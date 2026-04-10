import { X, Expand, Target, Code2 } from 'lucide-react'
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
  onToggleCode?: () => void
  showCode?: boolean
}

export function NodeInspector({
  node, edges, allNodes, onClose, onExpandNeighbors, onFocus, onToggleCode, showCode,
}: Props) {
  const connectedEdges = edges.filter((e) => e.src === node.id || e.dst === node.id)
  const nodeMap = new Map(allNodes.map((n) => [n.id, n]))

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 bg-zinc-800/30 px-4 py-2.5">
        <span
          className="inline-block h-2 w-2 shrink-0 rounded-full"
          style={{ backgroundColor: KIND_COLORS[node.kind] ?? '#52525b' }}
        />
        <span className="min-w-0 flex-1 truncate text-[12px] font-medium text-zinc-100" title={node.label}>
          {trimLabel(node.label)}
        </span>
        <button
          onClick={onClose}
          className="shrink-0 rounded p-1 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-400"
        >
          <X size={12} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Meta */}
        <div className="space-y-2 px-4 py-3">
          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-zinc-600">Kind</span>
            <span className="rounded bg-zinc-800/60 px-1.5 py-0.5 text-zinc-300">{node.kind}</span>
          </div>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-zinc-600">Centrality</span>
            <span className="tabular-nums text-zinc-300">{node.centrality.toFixed(4)}</span>
          </div>
          {node.namespace && (
            <div className="flex items-center gap-3 text-[11px]">
              <span className="text-zinc-600">Namespace</span>
              <span className="truncate text-zinc-400">{node.namespace}</span>
            </div>
          )}
          <div className="text-[10px] font-mono text-zinc-700 break-all">{node.id}</div>
        </div>

        {/* Actions */}
        <div className="flex gap-1.5 border-y border-zinc-800/30 px-4 py-2.5">
          <button
            onClick={() => onExpandNeighbors(node.id)}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-zinc-800/50 py-1.5 text-[11px] font-medium text-zinc-400 transition-colors hover:border-zinc-600 hover:text-zinc-200"
          >
            <Expand size={11} /> Expand
          </button>
          {onFocus && (
            <button
              onClick={() => onFocus(node.id)}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-zinc-800/50 py-1.5 text-[11px] font-medium text-zinc-400 transition-colors hover:border-zinc-600 hover:text-zinc-200"
            >
              <Target size={11} /> Focus
            </button>
          )}
          {onToggleCode && (
            <button
              onClick={onToggleCode}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-md border py-1.5 text-[11px] font-medium transition-colors ${
                showCode
                  ? 'border-violet-500/30 bg-violet-500/10 text-violet-400'
                  : 'border-zinc-800/50 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
              }`}
            >
              <Code2 size={11} /> Code
            </button>
          )}
        </div>

        {/* Connections */}
        <div className="px-4 py-3">
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
            Connections ({connectedEdges.length})
          </p>
          <div className="space-y-0.5">
            {connectedEdges.length === 0 ? (
              <p className="text-[11px] text-zinc-700">No connections</p>
            ) : (
              connectedEdges.slice(0, 30).map((edge) => {
                const targetId = edge.src === node.id ? edge.dst : edge.src
                const target = nodeMap.get(targetId)
                return (
                  <div
                    key={edge.id}
                    className="flex items-center gap-1.5 rounded py-1 pl-1 pr-2 text-[11px] transition-colors hover:bg-zinc-800/30"
                  >
                    <span className="shrink-0 rounded bg-zinc-800/60 px-1 py-0.5 text-[9px] font-medium text-zinc-500">
                      {edge.relationship}
                    </span>
                    <span className="min-w-0 truncate text-zinc-400">
                      {trimLabel(target?.label ?? targetId)}
                    </span>
                  </div>
                )
              })
            )}
            {connectedEdges.length > 30 && (
              <p className="pt-1 text-[10px] text-zinc-600">
                +{connectedEdges.length - 30} more
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
