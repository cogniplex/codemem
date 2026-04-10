import { useEffect, useRef, useMemo, useCallback, useState } from 'react'
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d'
import type { GraphNode, GraphEdge } from '../../api/types'
import { KIND_COLORS, EDGE_COLORS } from './constants'
import { trimLabel } from '../../utils/paths'

const CANVAS_OBJECT_MODE = () => 'replace' as const

const COMMUNITY_PALETTE = [
  '#a78bfa', '#22d3ee', '#34d399', '#fbbf24', '#f87171',
  '#60a5fa', '#c084fc', '#fb923c', '#2dd4bf', '#e879f9',
  '#818cf8', '#94a3b8', '#f472b6', '#a3e635', '#facc15',
]

function communityColor(id: number): string {
  return COMMUNITY_PALETTE[id % COMMUNITY_PALETTE.length]
}

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  communities?: Record<string, number> | null
  showCommunities: boolean
  showEdges: boolean
  onNodeClick?: (nodeId: string) => void
  onBackgroundClick?: () => void
  highlightNodeId?: string | null
  searchLabel?: string
  onLayoutRunning?: (running: boolean) => void
  activeRelationships?: Set<string> | null
  focusNodeId?: string | null
}

interface FGNode {
  id: string
  label: string
  kind: string
  centrality: number
  color: string
  size: number
  degree: number
  x?: number
  y?: number
}

interface FGLink {
  source: string
  target: string
  relationship: string
  weight: number
  color: string
}

export function SigmaGraph({
  nodes,
  edges,
  communities,
  showCommunities,
  showEdges,
  onNodeClick,
  onBackgroundClick,
  highlightNodeId,
  searchLabel,
  onLayoutRunning,
  activeRelationships,
  focusNodeId,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const fgRef = useRef<ForceGraphMethods<FGNode, FGLink> | undefined>(undefined)

  // Build graph data for force-graph
  const graphData = useMemo(() => {
    const nodeSet = new Set(nodes.map((n) => n.id))

    // Deduplicate edges
    const edgeSet = new Set<string>()
    const links: FGLink[] = []
    for (const edge of edges) {
      if (nodeSet.has(edge.src) && nodeSet.has(edge.dst) && edge.src !== edge.dst) {
        const pair = edge.src < edge.dst ? `${edge.src}:${edge.dst}` : `${edge.dst}:${edge.src}`
        const key = `${pair}:${edge.relationship}`
        if (!edgeSet.has(key)) {
          edgeSet.add(key)
          links.push({
            source: edge.src,
            target: edge.dst,
            relationship: edge.relationship,
            weight: edge.weight,
            color: EDGE_COLORS[edge.relationship] ?? '#52525b18',
          })
        }
      }
    }

    // Compute degree per node
    const degreeMap = new Map<string, number>()
    for (const link of links) {
      degreeMap.set(link.source, (degreeMap.get(link.source) ?? 0) + 1)
      degreeMap.set(link.target, (degreeMap.get(link.target) ?? 0) + 1)
    }

    const fgNodes: FGNode[] = nodes.map((node) => {
      const degree = degreeMap.get(node.id) ?? 0
      const size = 2 + Math.min(Math.log1p(degree) * 2, 12)
      return {
        id: node.id,
        label: trimLabel(node.label),
        kind: node.kind,
        centrality: node.centrality,
        color: KIND_COLORS[node.kind] ?? '#71717a',
        size,
        degree,
      }
    })

    return { nodes: fgNodes, links }
  }, [nodes, edges])

  // d3-force mutates link.source/target from string → object after simulation starts
  const linkNodeId = (node: string | FGNode) => typeof node === 'string' ? node : node.id

  // Highlight set: selected node + its neighbors
  const highlightSet = useMemo(() => {
    if (!highlightNodeId) return null
    const set = new Set<string>()
    set.add(highlightNodeId)
    for (const link of graphData.links) {
      const src = linkNodeId(link.source as string | FGNode)
      const tgt = linkNodeId(link.target as string | FGNode)
      if (src === highlightNodeId) set.add(tgt)
      if (tgt === highlightNodeId) set.add(src)
    }
    return set
  }, [highlightNodeId, graphData.links])

  // Configure d3 forces after mount
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return

    fg.d3Force('charge')?.strength((node: FGNode) => {
      // Stronger repulsion for high-degree hub nodes to spread clusters
      return -30 - (node.degree ?? 0) * 2
    })
    fg.d3Force('link')?.distance((link: FGLink) => {
      // Structural edges (CONTAINS) keep nodes close; weak edges stay loose
      if (link.relationship === 'CONTAINS' || link.relationship === 'PART_OF') return 20
      if (link.relationship === 'CALLS' || link.relationship === 'CO_CHANGED') return 80
      return 40
    }).strength((link: FGLink) => {
      if (link.relationship === 'CONTAINS' || link.relationship === 'PART_OF') return 0.8
      if (link.relationship === 'CALLS' || link.relationship === 'CO_CHANGED') return 0.05
      return 0.3
    })
    fg.d3Force('center')?.strength(0.05)

    fg.d3ReheatSimulation()
  }, [graphData])

  // Zoom to searched node
  useEffect(() => {
    if (!searchLabel || !fgRef.current) return
    const lowerSearch = searchLabel.toLowerCase()
    const found = graphData.nodes.find((n) => n.label.toLowerCase().includes(lowerSearch))
    if (found && found.x != null && found.y != null) {
      fgRef.current.centerAt(found.x, found.y, 300)
      fgRef.current.zoom(4, 300)
    }
  }, [searchLabel, graphData.nodes])

  // Recenter camera when entering/exiting focus mode
  useEffect(() => {
    if (!fgRef.current) return
    // Wait for layout to settle before zooming to fit
    const timer = setTimeout(() => {
      fgRef.current?.zoomToFit(400, 40)
    }, 500)
    return () => clearTimeout(timer)
  }, [focusNodeId])

  // Notify layout running state
  useEffect(() => {
    onLayoutRunning?.(true)
  }, [graphData, onLayoutRunning])

  const handleEngineStop = useCallback(() => {
    onLayoutRunning?.(false)
  }, [onLayoutRunning])

  const handleNodeClick = useCallback((node: FGNode) => {
    onNodeClick?.(node.id)
  }, [onNodeClick])

  // Node canvas renderer
  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- required by react-force-graph-2d callback signature
  const nodeCanvasObject = useCallback((node: FGNode, ctx: CanvasRenderingContext2D, _globalScale: number) => {
    const x = node.x ?? 0
    const y = node.y ?? 0
    const size = node.size

    // Determine color
    let color = node.color
    if (showCommunities && communities?.[node.id] != null) {
      color = communityColor(communities[node.id])
    }
    if (focusNodeId === node.id) {
      color = '#ffffff'
    }

    // Dim non-highlighted nodes
    let alpha = 1
    if (highlightSet) {
      if (node.id === highlightNodeId) {
        color = '#ffffff'
      } else if (!highlightSet.has(node.id)) {
        alpha = 0.15
      }
    }

    ctx.globalAlpha = alpha

    // Draw node circle
    ctx.beginPath()
    ctx.arc(x, y, size, 0, 2 * Math.PI)
    ctx.fillStyle = color
    ctx.fill()

    // No labels on canvas — tooltip on hover is sufficient

    ctx.globalAlpha = 1
  }, [highlightSet, highlightNodeId, showCommunities, communities, focusNodeId])

  // Link visibility filter
  const linkVisibility = useCallback((link: FGLink) => {
    if (!showEdges) return false
    const rel = link.relationship
    if (activeRelationships && !activeRelationships.has(rel)) return false

    // In highlight mode, only show edges connected to highlighted nodes
    if (highlightSet) {
      const src = linkNodeId(link.source as string | FGNode)
      const tgt = linkNodeId(link.target as string | FGNode)
      return highlightSet.has(src) && highlightSet.has(tgt)
    }
    return true
  }, [showEdges, activeRelationships, highlightSet])

  const linkWidth = useCallback((link: FGLink) => {
    const isWeak = link.relationship === 'CALLS' || link.relationship === 'CO_CHANGED'
    const isStructural = link.relationship === 'CONTAINS' || link.relationship === 'PART_OF'
    if (isWeak || isStructural) return 0.3
    return Math.max(0.5, link.weight * 1.5)
  }, [])

  const linkColor = useCallback((link: FGLink) => link.color, [])

  // Hover tooltip
  const nodeLabel = useCallback((node: FGNode) => {
    const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
    return `<div style="background:rgba(24,24,27,0.95);padding:8px 12px;border-radius:6px;color:#f4f4f5;font-size:12px;max-width:250px">
      <span style="background:${node.color}30;color:${node.color};padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">${esc(node.kind)}</span>
      <div style="margin-top:4px;font-weight:600;word-break:break-all">${esc(node.label)}</div>
      <div style="margin-top:4px;color:#a1a1aa;font-size:10px">${node.degree} connections · ${(node.centrality * 100).toFixed(1)}% centrality</div>
    </div>`
  }, [])

  // Container size
  const [dimensions] = useDimensions(containerRef)

  return (
    <div ref={containerRef} className="h-full w-full rounded-lg bg-zinc-950">
      {dimensions.width > 0 && (
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          nodeCanvasObject={nodeCanvasObject}
          nodeCanvasObjectMode={CANVAS_OBJECT_MODE}
          nodeVal={(node: FGNode) => node.size}
          onNodeClick={handleNodeClick}
          onBackgroundClick={onBackgroundClick}
          nodeLabel={nodeLabel}
          linkVisibility={linkVisibility}
          linkWidth={linkWidth}
          linkColor={linkColor}
          onEngineStop={handleEngineStop}
          linkDirectionalParticles={0}
          cooldownTicks={200}
          warmupTicks={50}
          backgroundColor="rgba(0,0,0,0)"
          enableNodeDrag={true}
          enableZoomInteraction={true}
          enablePanInteraction={true}
        />
      )}
    </div>
  )
}

// Hook to track container dimensions
function useDimensions(ref: React.RefObject<HTMLDivElement | null>) {
  const [dims, setDims] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const update = () => {
      setDims({ width: el.clientWidth, height: el.clientHeight })
    }
    update()

    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  // eslint-disable-next-line react-hooks/exhaustive-deps -- ref is stable
  }, [])

  return [dims] as const
}

