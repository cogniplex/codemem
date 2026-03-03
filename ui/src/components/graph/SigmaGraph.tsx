import { useEffect, useRef, useState, useCallback } from 'react'
import Graph from 'graphology'
import Sigma from 'sigma'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import type { GraphNode, GraphEdge } from '../../api/types'
import { KIND_COLORS, EDGE_COLORS } from './constants'
import { trimLabel } from '../../utils/paths'

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
  activeRelationships?: Set<string> | null
  focusNodeId?: string | null
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
  activeRelationships,
  focusNodeId,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)

  // Track graph instance in STATE so React properly detects changes
  // (refs don't trigger re-renders — this was the expand neighbors bug)
  const [graphInstance, setGraphInstance] = useState<Graph | null>(null)

  const notifyLayout = useCallback((running: boolean) => {
    onLayoutRunning?.(running)
  }, [onLayoutRunning])

  // Derive a common namespace prefix for path trimming
  const namespacePrefix = nodes[0]?.namespace ?? null

  // Build graph when data changes — community coloring is handled
  // by the node reducer below, not here, to avoid expensive rebuilds.
  useEffect(() => {
    const graph = new Graph()

    // First pass: add nodes with placeholder size
    for (const node of nodes) {
      graph.addNode(node.id, {
        label: trimLabel(node.label, namespacePrefix),
        size: 3, // placeholder — resized after edges are added
        color: KIND_COLORS[node.kind] ?? '#71717a',
        kind: node.kind,
        centrality: node.centrality,
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

    // Pre-compute layout synchronously — fully positions nodes before Sigma
    // renders, eliminating wiggle. No live supervisor needed after this.
    const nodeCount = graph.order
    if (nodeCount > 0) {
      // Scale iterations with graph size, but cap to avoid freezing
      const preIterations = nodeCount > 500 ? 300 : nodeCount > 100 ? 200 : 150
      forceAtlas2.assign(graph, {
        iterations: preIterations,
        settings: {
          gravity: 8,
          scalingRatio: nodeCount > 300 ? 30 : 15,
          slowDown: 10,
          barnesHutOptimize: true,
          barnesHutTheta: 0.5,
          strongGravityMode: true,
          linLogMode: false,
          adjustSizes: true,
        },
      })
    }

    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: graphInstance triggers Sigma re-init
    setGraphInstance(graph)

    return () => {
      graph.clear()
    }
  }, [nodes, edges, namespacePrefix])

  // Init Sigma renderer — depends on graphInstance STATE, not a ref
  useEffect(() => {
    if (!containerRef.current || !graphInstance) return

    const sigma = new Sigma(graphInstance, containerRef.current, {
      renderLabels: true,
      labelColor: { color: '#d4d4d8' },
      labelSize: 11,
      labelRenderedSizeThreshold: 6,
      defaultDrawNodeHover: (context, data) => {
        const attrs = data as Record<string, unknown>
        const label = (data.label ?? attrs.formerLabel) as string | undefined
        const kind = (attrs.kind as string) ?? ''
        const centrality = (attrs.centrality as number) ?? 0
        const degree = graphInstance.degree(data.key ?? '')

        const PAD = 8
        const LINE_H = 16
        const BADGE_H = 18
        const CARD_W = 200

        // Wrap label into lines
        const lines: string[] = []
        if (label) {
          context.font = '600 12px sans-serif'
          const words = label.split(/(?=[A-Z/_:.])|[\s]+/)
          let line = ''
          for (const word of words) {
            const test = line + word
            if (context.measureText(test).width > CARD_W - 2 * PAD && line) {
              lines.push(line)
              line = word
            } else {
              line = test
            }
          }
          if (line) lines.push(line)
          if (lines.length > 3) {
            lines.length = 3
            lines[2] = lines[2].slice(0, -3) + '...'
          }
        }

        const cardH = PAD + BADGE_H + 4 + lines.length * LINE_H + 4 + LINE_H + PAD
        const cardX = data.x + data.size + 6
        const cardY = data.y - cardH / 2

        // Shadow
        context.shadowOffsetX = 0
        context.shadowOffsetY = 2
        context.shadowBlur = 12
        context.shadowColor = 'rgba(0,0,0,0.5)'

        // Card background with rounded corners
        const r = 6
        context.beginPath()
        context.moveTo(cardX + r, cardY)
        context.lineTo(cardX + CARD_W - r, cardY)
        context.quadraticCurveTo(cardX + CARD_W, cardY, cardX + CARD_W, cardY + r)
        context.lineTo(cardX + CARD_W, cardY + cardH - r)
        context.quadraticCurveTo(cardX + CARD_W, cardY + cardH, cardX + CARD_W - r, cardY + cardH)
        context.lineTo(cardX + r, cardY + cardH)
        context.quadraticCurveTo(cardX, cardY + cardH, cardX, cardY + cardH - r)
        context.lineTo(cardX, cardY + r)
        context.quadraticCurveTo(cardX, cardY, cardX + r, cardY)
        context.closePath()
        context.fillStyle = 'rgba(24,24,27,0.95)'
        context.fill()
        context.shadowBlur = 0

        // Kind badge
        let badgeY = cardY + PAD
        if (kind) {
          const badgeColor = KIND_COLORS[kind] ?? '#71717a'
          context.font = '600 10px sans-serif'
          const badgeW = context.measureText(kind).width + 12
          const badgeR = 4
          context.beginPath()
          context.moveTo(cardX + PAD + badgeR, badgeY)
          context.lineTo(cardX + PAD + badgeW - badgeR, badgeY)
          context.quadraticCurveTo(cardX + PAD + badgeW, badgeY, cardX + PAD + badgeW, badgeY + badgeR)
          context.lineTo(cardX + PAD + badgeW, badgeY + BADGE_H - badgeR)
          context.quadraticCurveTo(cardX + PAD + badgeW, badgeY + BADGE_H, cardX + PAD + badgeW - badgeR, badgeY + BADGE_H)
          context.lineTo(cardX + PAD + badgeR, badgeY + BADGE_H)
          context.quadraticCurveTo(cardX + PAD, badgeY + BADGE_H, cardX + PAD, badgeY + BADGE_H - badgeR)
          context.lineTo(cardX + PAD, badgeY + badgeR)
          context.quadraticCurveTo(cardX + PAD, badgeY, cardX + PAD + badgeR, badgeY)
          context.closePath()
          context.fillStyle = badgeColor + '30'
          context.fill()
          context.fillStyle = badgeColor
          context.fillText(kind, cardX + PAD + 6, badgeY + 13)
          badgeY += BADGE_H + 4
        }

        // Label lines
        context.font = '600 12px sans-serif'
        context.fillStyle = '#f4f4f5'
        for (const line of lines) {
          context.fillText(line, cardX + PAD, badgeY + 12)
          badgeY += LINE_H
        }

        // Footer: degree + centrality
        badgeY += 4
        context.font = '400 10px sans-serif'
        context.fillStyle = '#a1a1aa'
        context.fillText(
          `${degree} connections  ·  ${(centrality * 100).toFixed(1)}% centrality`,
          cardX + PAD,
          badgeY + 10,
        )
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

    return () => {
      sigma.kill()
      sigmaRef.current = null
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

  // Helper: apply community color to a node if enabled
  const applyCommunityColor = useCallback((node: string, data: Record<string, unknown>) => {
    if (showCommunities && communities?.[node] != null) {
      return { ...data, color: communityColor(communities[node]) }
    }
    return data
  }, [showCommunities, communities])

  // Node + edge reducers: highlight, focus, community coloring, relationship filters, edge toggle
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
          const label = data.label ?? (data as Record<string, unknown>).formerLabel as string | undefined
          return { ...data, label, color: '#ffffff', zIndex: 10, size: (data.size ?? 5) * 1.5 }
        }
        if (connectedIds.has(node)) {
          const label = data.label ?? (data as Record<string, unknown>).formerLabel as string | undefined
          return applyCommunityColor(node, { ...data, label, zIndex: 5 })
        }
        return { ...data, color: '#3f3f46', zIndex: 0, label: null }
      })

      sigma.setSetting('edgeReducer', (edge, data) => {
        const src = graphInstance.source(edge)
        const dst = graphInstance.target(edge)
        const edgeLabel = (data as Record<string, unknown>).label as string | undefined
        if (activeRelationships && edgeLabel && !activeRelationships.has(edgeLabel)) {
          return { ...data, hidden: true }
        }
        if (connectedIds.has(src) && connectedIds.has(dst)) {
          return { ...data, size: Math.max((data.size ?? 1) * 1.5, 1.5), forceLabel: true }
        }
        return { ...data, color: '#27272a00', size: 0, hidden: true }
      })
    } else if (focusNodeId) {
      sigma.setSetting('nodeReducer', (node, data) => {
        if (node === focusNodeId) {
          const label = data.label ?? (data as Record<string, unknown>).formerLabel as string | undefined
          return { ...data, label, color: '#ffffff', zIndex: 10, size: (data.size ?? 5) * 1.8, borderColor: '#ffffff' }
        }
        const label = data.label ?? (data as Record<string, unknown>).formerLabel as string | undefined
        return applyCommunityColor(node, { ...data, label, zIndex: 5 })
      })

      sigma.setSetting('edgeReducer', (_edge, data) => {
        const edgeLabel = (data as Record<string, unknown>).label as string | undefined
        if (activeRelationships && edgeLabel && !activeRelationships.has(edgeLabel)) {
          return { ...data, hidden: true }
        }
        return { ...data, forceLabel: true }
      })
    } else {
      // Default: apply community coloring through reducer (no graph rebuild)
      sigma.setSetting('nodeReducer', (node, data) => applyCommunityColor(node, data))
      sigma.setSetting('edgeReducer', (_edge, data) => {
        if (!showEdges) {
          return { ...data, hidden: true }
        }
        const edgeLabel = (data as Record<string, unknown>).label as string | undefined
        if (activeRelationships && edgeLabel && !activeRelationships.has(edgeLabel)) {
          return { ...data, hidden: true }
        }
        return { ...data, label: null }
      })
    }

    sigma.refresh()
  }, [highlightNodeId, graphInstance, showEdges, activeRelationships, focusNodeId, applyCommunityColor])

  return (
    <div
      ref={containerRef}
      className="h-full w-full rounded-lg bg-zinc-950"
    />
  )
}
