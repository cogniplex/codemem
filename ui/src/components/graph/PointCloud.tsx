import { useRef, useState, useMemo, useCallback } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import { Search } from 'lucide-react'
import type { VectorPoint } from '../../api/types'
import { KIND_COLORS } from './constants'

const K_NEIGHBORS = 8

const TYPE_COLORS: Record<string, string> = {
  decision: '#c084fc',   // purple-400 — bright on black
  pattern: '#22d3ee',    // cyan-400
  preference: '#fbbf24', // amber-400
  style: '#34d399',      // emerald-400
  habit: '#fb7185',      // rose-400
  insight: '#60a5fa',    // blue-400
  context: '#a78bfa',    // violet-400
  chunk: '#94a3b8',      // slate-400 — visible grey
  ...KIND_COLORS,
}

/** Normalize PCA coordinates to fit within [-range, +range] */
function normalizePoints(points: VectorPoint[], range: number): VectorPoint[] {
  if (points.length === 0) return points
  let maxAbs = 0
  for (const p of points) {
    maxAbs = Math.max(maxAbs, Math.abs(p.x), Math.abs(p.y), Math.abs(p.z))
  }
  if (maxAbs < 1e-10) return points
  const scale = range / maxAbs
  return points.map((p) => ({ ...p, x: p.x * scale, y: p.y * scale, z: p.z * scale }))
}

/** Euclidean distance in PCA space */
function pcaDist(a: VectorPoint, b: VectorPoint): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  const dz = a.z - b.z
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/** Find K nearest neighbors in PCA space */
function findNearest(target: VectorPoint, all: VectorPoint[], k: number): { point: VectorPoint; dist: number }[] {
  const scored = all
    .filter((p) => p.id !== target.id)
    .map((p) => ({ point: p, dist: pcaDist(target, p) }))
  scored.sort((a, b) => a.dist - b.dist)
  return scored.slice(0, k)
}

interface CloudProps {
  points: VectorPoint[]
  selectedId: string | null
  neighborIds: Set<string>
  hiddenTypes: Set<string>
  highlightIds: Set<string> | null
  onSelect: (point: VectorPoint) => void
}

function Cloud({ points, selectedId, neighborIds, hiddenTypes, highlightIds, onSelect }: CloudProps) {
  const pointsRef = useRef<THREE.Points>(null)

  const visiblePoints = useMemo(
    () => points.filter((p) => !hiddenTypes.has(p.memory_type)),
    [points, hiddenTypes],
  )

  // Build position + color + alpha buffers
  const { positions, colors, alphas, sizes } = useMemo(() => {
    const pos = new Float32Array(visiblePoints.length * 3)
    const col = new Float32Array(visiblePoints.length * 3)
    const alp = new Float32Array(visiblePoints.length)
    const sz = new Float32Array(visiblePoints.length)
    const tempColor = new THREE.Color()

    for (let i = 0; i < visiblePoints.length; i++) {
      const p = visiblePoints[i]
      pos[i * 3] = p.x
      pos[i * 3 + 1] = p.y
      pos[i * 3 + 2] = p.z

      const isSelected = p.id === selectedId
      const isNeighbor = neighborIds.has(p.id)
      const isHighlighted = highlightIds ? highlightIds.has(p.id) : true

      let color = TYPE_COLORS[p.memory_type] ?? '#71717a'
      let alpha = 0.85

      if (isSelected) {
        color = '#ffffff'
      } else if (selectedId && !isNeighbor) {
        alpha = 0.25
      } else if (highlightIds && !isHighlighted) {
        alpha = 0.25
      }

      tempColor.set(color)
      col[i * 3] = tempColor.r
      col[i * 3 + 1] = tempColor.g
      col[i * 3 + 2] = tempColor.b
      alp[i] = alpha
      sz[i] = isSelected ? 6 : isNeighbor ? 4 : 1.5 + p.importance * 2
    }
    return { positions: pos, colors: col, alphas: alp, sizes: sz }
  }, [visiblePoints, selectedId, neighborIds, highlightIds])

  // Raycasting for click
  const handlePointerDown = (e: { index?: number }) => {
    if (e.index !== undefined && e.index < visiblePoints.length) {
      onSelect(visiblePoints[e.index])
    }
  }

  return (
    <points
      ref={pointsRef}
      onPointerDown={handlePointerDown}
    >
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        <bufferAttribute attach="attributes-aAlpha" args={[alphas, 1]} />
        <bufferAttribute attach="attributes-size" args={[sizes, 1]} />
      </bufferGeometry>
      <shaderMaterial
        vertexColors
        transparent
        depthWrite={false}
        vertexShader={`
          attribute float size;
          attribute float aAlpha;
          varying vec3 vColor;
          varying float vAlpha;
          void main() {
            vColor = color;
            vAlpha = aAlpha;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = size * (40.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
          }
        `}
        fragmentShader={`
          varying vec3 vColor;
          varying float vAlpha;
          void main() {
            vec2 center = gl_PointCoord - vec2(0.5);
            if (dot(center, center) > 0.25) discard;
            float dist = length(center) * 2.0;
            float edge = 1.0 - smoothstep(0.7, 1.0, dist);
            gl_FragColor = vec4(vColor, vAlpha * edge);
          }
        `}
      />
    </points>
  )
}

/** Lines connecting selected point to its nearest neighbors */
function NeighborLines({ selected, neighbors }: { selected: VectorPoint; neighbors: VectorPoint[] }) {
  return (
    <>
      {neighbors.map((n) => (
        <Line
          key={n.id}
          points={[
            [selected.x, selected.y, selected.z],
            [n.x, n.y, n.z],
          ]}
          color="#8b5cf6"
          lineWidth={0.5}
          transparent
          opacity={0.4}
        />
      ))}
    </>
  )
}


interface Props {
  points: VectorPoint[]
}

export function PointCloud({ points: rawPoints }: Props) {
  const [selectedPoint, setSelectedPoint] = useState<VectorPoint | null>(null)
  const [search, setSearch] = useState('')
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set())
  const [showLines, setShowLines] = useState(true)

  // Normalize coordinates
  const points = useMemo(() => normalizePoints(rawPoints, 20), [rawPoints])

  // Nearest neighbors of selected point
  const nearest = useMemo(() => {
    if (!selectedPoint) return []
    const normalizedSelected = points.find((p) => p.id === selectedPoint.id)
    if (!normalizedSelected) return []
    return findNearest(normalizedSelected, points, K_NEIGHBORS)
  }, [selectedPoint, points])

  const neighborIds = useMemo(
    () => new Set(nearest.map((n) => n.point.id)),
    [nearest],
  )

  // Search: find matching points
  const searchResults = useMemo(() => {
    if (!search || search.length < 2) return null
    const lower = search.toLowerCase()
    const matches = points.filter((p) => p.label.toLowerCase().includes(lower))
    return new Set(matches.map((p) => p.id))
  }, [search, points])

  // Unique types for legend
  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const p of points) {
      counts[p.memory_type] = (counts[p.memory_type] ?? 0) + 1
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1])
  }, [points])

  const toggleType = useCallback((type: string) => {
    setHiddenTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }, [])

  const handleSelect = useCallback((point: VectorPoint) => {
    setSelectedPoint((prev) => prev?.id === point.id ? null : point)
  }, [])

  const selectedNormalized = selectedPoint ? points.find((p) => p.id === selectedPoint.id) ?? null : null

  // Max distance for similarity % calculation
  const maxDist = nearest.length > 0 ? nearest[nearest.length - 1].dist : 1

  return (
    <div className="relative flex h-full w-full">
      {/* 3D Canvas */}
      <div className="flex-1">
        <Canvas camera={{ position: [0, 15, 45], fov: 55 }} onPointerMissed={() => setSelectedPoint(null)}>
          <color attach="background" args={['#09090b']} />
          <fog attach="fog" args={['#09090b', 40, 80]} />
          <Cloud
            points={points}
            selectedId={selectedPoint?.id ?? null}
            neighborIds={neighborIds}
            hiddenTypes={hiddenTypes}
            highlightIds={searchResults}
            onSelect={handleSelect}
          />
          {showLines && selectedNormalized && nearest.length > 0 && (
            <NeighborLines
              selected={selectedNormalized}
              neighbors={nearest.map((n) => n.point)}
            />
          )}
          <OrbitControls
            makeDefault
            target={[0, 0, 0]}
            enableDamping
            dampingFactor={0.12}
            rotateSpeed={0.5}
            zoomSpeed={0.8}
            enablePan={false}
            minDistance={5}
            maxDistance={80}
            minPolarAngle={Math.PI * 0.15}
            maxPolarAngle={Math.PI * 0.85}
            autoRotate={!selectedPoint}
            autoRotateSpeed={0.5}
          />

          {selectedNormalized && (
            <Html position={[selectedNormalized.x, selectedNormalized.y + 1.5, selectedNormalized.z]}>
              <div className="pointer-events-none w-52 rounded-lg border border-zinc-700 bg-zinc-900/95 p-2.5 text-xs shadow-lg">
                <p className="font-medium text-zinc-200 break-words">{selectedNormalized.label}</p>
                <div className="mt-1 flex items-center gap-2 text-zinc-400">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: TYPE_COLORS[selectedNormalized.memory_type] ?? '#71717a' }}
                  />
                  {selectedNormalized.memory_type}
                  <span className="text-zinc-500">
                    {(selectedNormalized.importance * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </Html>
          )}
        </Canvas>
      </div>

      {/* Right panel: search + neighbors */}
      <div className="flex w-72 shrink-0 flex-col border-l border-zinc-800 bg-zinc-950">
        {/* Stats */}
        <div className="border-b border-zinc-800 px-3 py-2 text-xs text-zinc-500">
          {points.length.toLocaleString()} points · 768-dim · PCA → 3D
        </div>

        {/* Search */}
        <div className="border-b border-zinc-800 p-3">
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search embeddings..."
              className="w-full rounded-md border border-zinc-700 bg-zinc-800 py-1.5 pl-8 pr-3 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500 focus:outline-none"
            />
          </div>
          {searchResults && (
            <p className="mt-1.5 text-xs text-zinc-500">
              {searchResults.size} matches highlighted
            </p>
          )}
        </div>

        {/* Selected point + neighbors */}
        {selectedPoint && (
          <div className="flex-1 overflow-y-auto border-b border-zinc-800">
            <div className="border-b border-zinc-800 p-3">
              <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Selected</p>
              <p className="mt-1 text-sm font-medium text-zinc-200 break-words">{selectedPoint.label}</p>
              <div className="mt-1 flex items-center gap-2 text-xs text-zinc-400">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: TYPE_COLORS[selectedPoint.memory_type] ?? '#71717a' }}
                />
                {selectedPoint.memory_type}
                <span>· {(selectedPoint.importance * 100).toFixed(0)}% importance</span>
              </div>
            </div>

            {nearest.length > 0 && (
              <div className="p-3">
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                    Nearest Neighbors
                  </p>
                  <button
                    onClick={() => setShowLines((v) => !v)}
                    className={`text-[10px] ${showLines ? 'text-violet-400' : 'text-zinc-500'}`}
                  >
                    {showLines ? 'Lines on' : 'Lines off'}
                  </button>
                </div>
                <div className="space-y-1.5">
                  {nearest.map(({ point, dist }, i) => {
                    // 1.5x inflator so the farthest neighbor shows ~33% instead of 0%
                    const similarity = Math.max(0, 1 - dist / (maxDist * 1.5))
                    return (
                      <button
                        key={point.id}
                        onClick={() => handleSelect(point)}
                        className="w-full rounded-md border border-zinc-800 bg-zinc-900 p-2 text-left transition-colors hover:border-zinc-700"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <p className="text-xs text-zinc-300 break-words line-clamp-2">{point.label}</p>
                          <span className="shrink-0 text-[10px] font-mono text-zinc-500">
                            {(similarity * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="mt-1 flex items-center gap-1.5">
                          <span
                            className="inline-block h-1.5 w-1.5 rounded-full"
                            style={{ backgroundColor: TYPE_COLORS[point.memory_type] ?? '#71717a' }}
                          />
                          <span className="text-[10px] text-zinc-500">{point.memory_type}</span>
                          <span className="text-[10px] text-zinc-600">#{i + 1}</span>
                        </div>
                        {/* Similarity bar */}
                        <div className="mt-1.5 h-0.5 w-full rounded-full bg-zinc-800">
                          <div
                            className="h-full rounded-full bg-violet-500/60"
                            style={{ width: `${similarity * 100}%` }}
                          />
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {!selectedPoint && (
          <div className="flex flex-1 items-center justify-center p-4">
            <p className="text-center text-xs text-zinc-500">
              Click a point to see its nearest neighbors in embedding space
            </p>
          </div>
        )}

        {/* Type legend — clickable to filter */}
        <div className="border-t border-zinc-800 p-3">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
            Types (click to filter)
          </p>
          <div className="flex flex-wrap gap-1.5">
            {typeCounts.map(([type, count]) => {
              const hidden = hiddenTypes.has(type)
              return (
                <button
                  key={type}
                  onClick={() => toggleType(type)}
                  className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-opacity ${
                    hidden ? 'opacity-30' : 'opacity-100'
                  } hover:opacity-80`}
                >
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: TYPE_COLORS[type] ?? '#71717a' }}
                  />
                  <span className="text-zinc-400">{type}</span>
                  <span className="text-zinc-600">{count}</span>
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
