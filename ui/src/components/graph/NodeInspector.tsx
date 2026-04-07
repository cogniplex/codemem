import { X, Expand, Target } from 'lucide-react'
import type { GraphNode, GraphEdge } from '../../api/types'
import { KIND_COLORS } from './constants'
import { trimLabel } from '../../utils/paths'

interface Props {
  node: GraphNode
  edges: GraphEdge[]
  allNodes: GraphNode[]
  onClose: () => void
  onExpandNeighbors: (nodeId: string) => void
  onFocus?: (nodeId: string) => void
}

export function NodeInspector({ node, edges, allNodes, onClose, onExpandNeighbors, onFocus }: Props) {
  const connectedEdges = edges.filter((e) => e.src === node.id || e.dst === node.id)
  const nodeMap = new Map(allNodes.map((n) => [n.id, n]))

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
        <span className="text-sm font-medium text-zinc-200">Details</span>
        <button
          onClick={onClose}
          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          <X size={14} />
        </button>
      </div>

      {/* Node details */}
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        <div>
          <p className="text-xs text-zinc-500">Label</p>
          <p className="text-sm font-medium text-zinc-200 break-words">{trimLabel(node.label)}</p>
        </div>

        <div className="flex gap-4">
          <div>
            <p className="text-xs text-zinc-500">Kind</p>
            <span className="inline-flex items-center gap-1.5 rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: KIND_COLORS[node.kind] ?? '#71717a' }}
              />
              {node.kind}
            </span>
          </div>
          <div>
            <p className="text-xs text-zinc-500">Centrality</p>
            <p className="text-sm tabular-nums text-zinc-300">{node.centrality.toFixed(4)}</p>
          </div>
        </div>

        {node.namespace && (
          <div>
            <p className="text-xs text-zinc-500">Namespace</p>
            <p className="truncate text-sm text-zinc-300">{node.namespace}</p>
          </div>
        )}

        <div>
          <p className="text-xs text-zinc-500">ID</p>
          <p className="truncate text-xs font-mono text-zinc-500">{node.id}</p>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => onExpandNeighbors(node.id)}
            className="flex flex-1 items-center justify-center gap-2 rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-300 hover:border-violet-500 hover:text-violet-300"
          >
            <Expand size={14} />
            Expand
          </button>
          {onFocus && (
            <button
              onClick={() => onFocus(node.id)}
              className="flex flex-1 items-center justify-center gap-2 rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-300 hover:border-violet-500 hover:text-violet-300"
            >
              <Target size={14} />
              Focus
            </button>
          )}
        </div>

        {/* Connected edges */}
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-400">
            Connections ({connectedEdges.length})
          </p>
          <div className="max-h-64 space-y-1 overflow-y-auto">
            {connectedEdges.length === 0 && (
              <p className="text-xs text-zinc-600">No connections</p>
            )}
            {connectedEdges.map((edge) => {
              const targetId = edge.src === node.id ? edge.dst : edge.src
              const target = nodeMap.get(targetId)
              return (
                <div
                  key={edge.id}
                  className="flex min-w-0 items-center gap-2 rounded bg-zinc-850 px-2 py-1 text-xs"
                >
                  <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">
                    {edge.relationship}
                  </span>
                  <span className="truncate text-zinc-300">{trimLabel(target?.label ?? targetId)}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
