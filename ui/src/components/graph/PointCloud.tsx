import { useRef, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import type { VectorPoint } from '../../api/types'
import { KIND_COLORS } from './constants'

const TYPE_COLORS: Record<string, string> = {
  decision: '#8b5cf6',
  pattern: '#06b6d4',
  preference: '#f59e0b',
  style: '#10b981',
  habit: '#ef4444',
  insight: '#3b82f6',
  context: '#6366f1',
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

interface CloudProps {
  points: VectorPoint[]
  onHover: (point: VectorPoint | null) => void
}

function Cloud({ points, onHover }: CloudProps) {
  const groupRef = useRef<THREE.Group>(null)
  const pointsRef = useRef<THREE.Points>(null)

  // Build position + color buffers
  const { positions, colors, sizes } = useMemo(() => {
    const pos = new Float32Array(points.length * 3)
    const col = new Float32Array(points.length * 3)
    const sz = new Float32Array(points.length)
    const tempColor = new THREE.Color()

    for (let i = 0; i < points.length; i++) {
      const p = points[i]
      pos[i * 3] = p.x
      pos[i * 3 + 1] = p.y
      pos[i * 3 + 2] = p.z
      tempColor.set(TYPE_COLORS[p.memory_type] ?? '#71717a')
      col[i * 3] = tempColor.r
      col[i * 3 + 1] = tempColor.g
      col[i * 3 + 2] = tempColor.b
      sz[i] = 2 + p.importance * 4
    }
    return { positions: pos, colors: col, sizes: sz }
  }, [points])

  // Slow rotation
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.03
    }
  })

  // Raycasting for hover — use pointer events on the points object
  const handlePointerMove = (e: { index?: number }) => {
    if (e.index !== undefined && e.index < points.length) {
      onHover(points[e.index])
    }
  }

  return (
    <group ref={groupRef}>
      <points
        ref={pointsRef}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => onHover(null)}
      >
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[colors, 3]}
          />
          <bufferAttribute
            attach="attributes-size"
            args={[sizes, 1]}
          />
        </bufferGeometry>
        <pointsMaterial
          vertexColors
          size={0.6}
          sizeAttenuation
          transparent
          opacity={0.85}
          depthWrite={false}
        />
      </points>
    </group>
  )
}

interface Props {
  points: VectorPoint[]
}

export function PointCloud({ points: rawPoints }: Props) {
  const [hoveredPoint, setHoveredPoint] = useState<VectorPoint | null>(null)

  // Normalize coordinates to [-20, +20] range so camera can see them
  const points = useMemo(() => normalizePoints(rawPoints, 20), [rawPoints])

  // Unique memory types for legend
  const types = useMemo(() => {
    const set = new Set(points.map((p) => p.memory_type))
    return [...set].sort()
  }, [points])

  return (
    <div className="relative h-full w-full">
      <Canvas camera={{ position: [35, 25, 35], fov: 50 }}>
        <color attach="background" args={['#09090b']} />
        <Cloud points={points} onHover={setHoveredPoint} />
        <OrbitControls enableDamping dampingFactor={0.05} />

        {hoveredPoint && (
          <Html position={[hoveredPoint.x, hoveredPoint.y + 1.5, hoveredPoint.z]}>
            <div className="pointer-events-none w-52 rounded-lg border border-zinc-700 bg-zinc-900/95 p-2.5 text-xs shadow-lg">
              <p className="font-medium text-zinc-200 break-words">{hoveredPoint.label}</p>
              <div className="mt-1 flex items-center gap-2 text-zinc-400">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: TYPE_COLORS[hoveredPoint.memory_type] ?? '#71717a' }}
                />
                {hoveredPoint.memory_type}
                <span className="text-zinc-500">
                  {(hoveredPoint.importance * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </Html>
        )}
      </Canvas>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 rounded-lg border border-zinc-800 bg-zinc-900/90 p-3 backdrop-blur-sm">
        <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
          Types
        </p>
        <div className="flex flex-wrap gap-2">
          {types.map((t) => (
            <div key={t} className="flex items-center gap-1.5">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: TYPE_COLORS[t] ?? '#71717a' }}
              />
              <span className="text-xs text-zinc-400">{t}</span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[10px] text-zinc-500">
          {points.length} points · Size = importance
        </p>
      </div>
    </div>
  )
}
