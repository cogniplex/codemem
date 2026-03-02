import { useEffect, useRef, useState, useCallback } from 'react'
import Graph from 'graphology'
import Sigma from 'sigma'
import FA2LayoutSupervisor from 'graphology-layout-forceatlas2/worker'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import type { GraphNode, GraphEdge } from '../../api/types'

export const KIND_COLORS: Record<string, string> = {
  function: '#8b5cf6',   // violet-500
  method: '#a78bfa',     // violet-400
  class: '#06b6d4',      // cyan-500
  file: '#10b981',       // emerald-500
  module: '#f59e0b',     // amber-500
  package: '#d97706',    // amber-600
  variable: '#ef4444',   // red-500
  type: '#3b82f6',       // blue-500
  interface: '#a855f7',  // purple-500
  trait: '#f97316',      // orange-500
  struct: '#14b8a6',     // teal-500
  enum: '#d946ef',       // fuchsia-500
  memory: '#6366f1',     // indigo-500
  constant: '#facc15',   // yellow-400
  endpoint: '#f43f5e',   // rose-500
  test: '#64748b',       // slate-500
}

const EDGE_COLORS: Record<string, string> = {
  // Structural — blue/cyan
  CONTAINS: '#3b82f650', PART_OF: '#3b82f650', IMPORTS: '#06b6d4a0',
  // Execution — subtle so CALLS (97% of edges) doesn't blind you
  CALLS: '#52525b30', EXTENDS: '#34d39960', IMPLEMENTS: '#22c55ea0', INHERITS: '#4ade8060',
  // Semantic — purple/violet
  RELATES_TO: '#8b5cf660', SIMILAR_TO: '#a78bfa60', SHARES_THEME: '#c084fc60',
  EXPLAINS: '#a78bfa60', EXEMPLIFIES: '#c084fc60', SUMMARIZES: '#8b5cf660',
  // Temporal — amber/orange
  LEADS_TO: '#f59e0b60', PRECEDED_BY: '#fbbf2460', EVOLVED_INTO: '#fb923c60', DERIVED_FROM: '#f9731660',
  // Dependency — teal
  DEPENDS_ON: '#14b8a660',
  // Negative — red/rose
  CONTRADICTS: '#ef444480', INVALIDATED_BY: '#f8717180', BLOCKS: '#dc262680', SUPERSEDES: '#fb718580',
  // Reinforcement — lime
  REINFORCES: '#a3e63580',
}

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
  highlightNodeId?: string | null
  searchLabel?: string
  onLayoutRunning?: (running: boolean) => void
}

export function SigmaGraph({
  nodes,
  edges,
  communities,
  showCommunities,
  showEdges,
  onNodeClick,
  highlightNodeId,
  searchLabel,
  onLayoutRunning,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)
  const layoutRef = useRef<FA2LayoutSupervisor | null>(null)

  // Track graph instance in STATE so React properly detects changes
  // (refs don't trigger re-renders — this was the expand neighbors bug)
  const [graphInstance, setGraphInstance] = useState<Graph | null>(null)

  const notifyLayout = useCallback((running: boolean) => {
    onLayoutRunning?.(running)
  }, [onLayoutRunning])

  // Build graph when data changes
  useEffect(() => {
    const graph = new Graph()

    // First pass: add nodes with placeholder size
    for (const node of nodes) {
      const color =
        showCommunities && communities?.[node.id] != null
          ? communityColor(communities[node.id])
          : KIND_COLORS[node.kind] ?? '#71717a'

      graph.addNode(node.id, {
        label: node.label,
        size: 3, // placeholder — resized after edges are added
        color,
        x: Math.random() * 100,
        y: Math.random() * 100,
      })
    }

    // Add edges
    const nodeSet = new Set(nodes.map((n) => n.id))
    const edgeSet = new Set<string>()
    for (const edge of edges) {
      if (nodeSet.has(edge.src) && nodeSet.has(edge.dst) && edge.src !== edge.dst) {
        const key = edge.src < edge.dst ? `${edge.src}:${edge.dst}` : `${edge.dst}:${edge.src}`
        if (!edgeSet.has(key)) {
          edgeSet.add(key)
          const edgeColor = EDGE_COLORS[edge.relationship] ?? '#52525b18'
          const isCommon = edge.relationship === 'CALLS'
          graph.addEdgeWithKey(key, edge.src, edge.dst, {
            weight: edge.weight,
            label: edge.relationship,
            size: isCommon ? 0.2 : Math.max(0.5, edge.weight * 1.5),
            color: edgeColor,
          })
        }
      }
    }

    // Second pass: size nodes by degree (connections) for structural importance
    graph.forEachNode((nodeId, attrs) => {
      const degree = graph.degree(nodeId)
      // Log-scaled: isolated nodes = 3, 5 connections = 6, 20 = 10, 100+ = 18
      const size = 3 + Math.min(Math.log1p(degree) * 3, 18)
      graph.setNodeAttribute(nodeId, 'size', size)
      // Only show labels for well-connected nodes (reduces clutter)
      if (degree < 3) {
        graph.setNodeAttribute(nodeId, 'label', null)
        graph.setNodeAttribute(nodeId, 'formerLabel', attrs.label)
      }
    })

    // Pre-compute layout synchronously so nodes are fully positioned before
    // Sigma renders — eliminates the initial chaotic wiggle entirely.
    const nodeCount = graph.order
    if (nodeCount > 0) {
      const preIterations = nodeCount > 500 ? 300 : nodeCount > 100 ? 200 : 100
      forceAtlas2.assign(graph, {
        iterations: preIterations,
        settings: {
          gravity: 5,
          scalingRatio: nodeCount > 300 ? 30 : 15,
          slowDown: 5,
          barnesHutOptimize: nodeCount > 100,
          barnesHutTheta: 0.5,
          strongGravityMode: true,
          linLogMode: false,
          adjustSizes: true,
        },
      })
    }

    setGraphInstance(graph)

    return () => {
      graph.clear()
    }
  }, [nodes, edges, communities, showCommunities])

  // Init Sigma renderer — depends on graphInstance STATE, not a ref
  useEffect(() => {
    if (!containerRef.current || !graphInstance) return

    const sigma = new Sigma(graphInstance, containerRef.current, {
      renderLabels: true,
      labelColor: { color: '#d4d4d8' },
      labelSize: 11,
      labelRenderedSizeThreshold: 6,
      defaultDrawNodeHover: (context, data, settings) => {
        // Show the full label on hover (even for nodes whose label was hidden)
        const label = data.label ?? (data as Record<string, unknown>).formerLabel as string | undefined
        const size = settings.labelSize
        const font = settings.labelFont
        const weight = settings.labelWeight
        context.font = `${weight} ${size}px ${font}`

        context.fillStyle = '#18181bf0'
        context.shadowOffsetX = 0
        context.shadowOffsetY = 0
        context.shadowBlur = 8
        context.shadowColor = '#000'

        const PADDING = 4
        if (typeof label === 'string') {
          const textWidth = context.measureText(label).width
          const boxWidth = Math.round(textWidth + 8)
          const boxHeight = Math.round(size + 2 * PADDING)
          const radius = Math.max(data.size, size / 2) + PADDING

          const angleRadian = Math.asin(boxHeight / 2 / radius)
          const xDeltaCoord = Math.sqrt(Math.abs(radius ** 2 - (boxHeight / 2) ** 2))

          context.beginPath()
          context.moveTo(data.x + xDeltaCoord, data.y + boxHeight / 2)
          context.lineTo(data.x + radius + boxWidth, data.y + boxHeight / 2)
          context.lineTo(data.x + radius + boxWidth, data.y - boxHeight / 2)
          context.lineTo(data.x + xDeltaCoord, data.y - boxHeight / 2)
          context.arc(data.x, data.y, radius, angleRadian, -angleRadian)
          context.closePath()
          context.fill()
        } else {
          context.beginPath()
          context.arc(data.x, data.y, data.size + PADDING, 0, Math.PI * 2)
          context.closePath()
          context.fill()
        }

        context.shadowBlur = 0

        if (label) {
          context.fillStyle = '#f4f4f5'
          context.fillText(label, data.x + data.size + 3, data.y + size / 3)
        }
      },
      renderEdgeLabels: true,
      edgeLabelColor: { color: '#a1a1aa' },
      edgeLabelSize: 10,
      defaultEdgeColor: '#52525b18',
      defaultNodeColor: '#71717a',
    })
    sigmaRef.current = sigma

    sigma.on('clickNode', ({ node }) => {
      onNodeClick?.(node)
    })

    // Layout is fully pre-computed synchronously — no live FA2 needed.
    // Keep the supervisor available for on-demand use (e.g. after expanding neighbors).
    const nodeCount = graphInstance.order
    const layout = new FA2LayoutSupervisor(graphInstance, {
      settings: {
        gravity: 5,
        scalingRatio: nodeCount > 300 ? 30 : 15,
        slowDown: 20,
        barnesHutOptimize: nodeCount > 100,
        barnesHutTheta: 0.5,
        strongGravityMode: true,
        linLogMode: false,
        adjustSizes: true,
      },
    })
    layoutRef.current = layout

    return () => {
      layout.kill()
      sigma.kill()
      sigmaRef.current = null
      layoutRef.current = null
    }
  }, [graphInstance, onNodeClick, nodes.length, notifyLayout])

  // Highlight searched node
  useEffect(() => {
    const sigma = sigmaRef.current
    if (!sigma || !graphInstance || !searchLabel) return

    const lowerSearch = searchLabel.toLowerCase()
    let found: string | null = null
    graphInstance.forEachNode((key, attrs) => {
      if (!found && attrs.label?.toLowerCase().includes(lowerSearch)) {
        found = key
      }
    })

    if (found) {
      const attrs = graphInstance.getNodeAttributes(found)
      sigma.getCamera().animate({ x: attrs.x, y: attrs.y, ratio: 0.3 }, { duration: 300 })
    }
  }, [searchLabel, graphInstance])

  // Highlight selected node — dim others for visual focus
  useEffect(() => {
    const sigma = sigmaRef.current
    if (!sigma || !graphInstance) return

    if (highlightNodeId) {
      const connectedIds = new Set<string>()
      connectedIds.add(highlightNodeId)
      graphInstance.forEachEdge(highlightNodeId, (_edge, _attrs, src, dst) => {
        connectedIds.add(src)
        connectedIds.add(dst)
      })

      sigma.setSetting('nodeReducer', (node, data) => {
        if (node === highlightNodeId) {
          // Restore label for highlighted node
          const label = data.label ?? (data as Record<string, unknown>).formerLabel as string | undefined
          return { ...data, label, color: '#ffffff', zIndex: 10, size: (data.size ?? 5) * 1.5 }
        }
        if (connectedIds.has(node)) {
          const label = data.label ?? (data as Record<string, unknown>).formerLabel as string | undefined
          return { ...data, label, zIndex: 5 }
        }
        return { ...data, color: '#3f3f46', zIndex: 0, label: null }
      })

      sigma.setSetting('edgeReducer', (edge, data) => {
        const src = graphInstance.source(edge)
        const dst = graphInstance.target(edge)
        if (connectedIds.has(src) && connectedIds.has(dst)) {
          return { ...data, size: Math.max((data.size ?? 1) * 1.5, 1.5), forceLabel: true }
        }
        return { ...data, color: '#27272a00', size: 0, hidden: true }
      })
    } else {
      sigma.setSetting('nodeReducer', (_node, data) => data)
      // When no node is selected: hide or show edges based on toggle
      sigma.setSetting('edgeReducer', (_edge, data) => {
        if (!showEdges) {
          return { ...data, hidden: true }
        }
        return { ...data, label: null }
      })
    }

    sigma.refresh()
  }, [highlightNodeId, graphInstance, showEdges])

  return (
    <div
      ref={containerRef}
      className="h-full w-full rounded-lg bg-zinc-950"
    />
  )
}
