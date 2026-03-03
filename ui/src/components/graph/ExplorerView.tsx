import { useState, useMemo } from 'react'
import { Search, Loader2, ChevronLeft, ChevronRight } from 'lucide-react'
import { useGraphBrowse, useNeighbors } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { KindFilterChips } from './KindFilterChips'
import { BrowseNodeCard } from './BrowseNodeCard'
import { SigmaGraph } from './SigmaGraph'
import { EDGE_COLORS } from './constants'
import { trimLabel, trimNamespace } from '../../utils/paths'
import type { BrowseNodeItem } from '../../api/types'

const PAGE_SIZE = 30

export function ExplorerView() {
  const namespace = useNamespaceStore((s) => s.active)
  const [search, setSearch] = useState('')
  const [activeKind, setActiveKind] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const [selectedNode, setSelectedNode] = useState<BrowseNodeItem | null>(null)

  const params = useMemo(
    () => ({
      namespace: namespace ?? undefined,
      kind: activeKind ?? undefined,
      q: search || undefined,
      offset: page * PAGE_SIZE,
      limit: PAGE_SIZE,
    }),
    [namespace, activeKind, search, page],
  )

  const { data, isLoading } = useGraphBrowse(params)
  const { data: neighborData, isLoading: neighborLoading, isError: neighborError } = useNeighbors(selectedNode?.id ?? '', 1)

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0

  return (
    <div className="flex h-full gap-0">
      {/* Left panel: list */}
      <div className="flex w-96 shrink-0 flex-col border-r border-zinc-800">
        {/* Search */}
        <div className="border-b border-zinc-800 p-3">
          <div className="relative">
            <Search
              size={14}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500"
            />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setPage(0)
              }}
              placeholder="Search nodes..."
              className="w-full rounded-md border border-zinc-700 bg-zinc-800 py-1.5 pl-8 pr-3 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Kind chips */}
        {data?.kinds && (
          <div className="border-b border-zinc-800 p-3">
            <KindFilterChips
              kinds={data.kinds}
              activeKind={activeKind}
              onSelect={(k) => {
                setActiveKind(k)
                setPage(0)
              }}
            />
          </div>
        )}

        {/* Stats bar */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2 text-xs text-zinc-500">
          <span>{data?.total ?? 0} nodes</span>
          <span>{data?.edge_count ?? 0} edges</span>
        </div>

        {/* Node list */}
        <div className="flex-1 overflow-y-auto p-2">
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 size={16} className="animate-spin text-zinc-500" />
            </div>
          )}
          {data?.nodes.map((node) => (
            <div key={node.id} className="mb-1.5">
              <BrowseNodeCard
                node={node}
                selected={selectedNode?.id === node.id}
                onClick={() => setSelectedNode(node)}
              />
            </div>
          ))}
          {data && data.nodes.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-500">No nodes match filters</p>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-zinc-800 px-3 py-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="rounded p-1 text-zinc-400 hover:text-zinc-200 disabled:opacity-30"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-xs text-zinc-500">
              {page + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="rounded p-1 text-zinc-400 hover:text-zinc-200 disabled:opacity-30"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        )}
      </div>

      {/* Right panel: detail */}
      <div className="flex min-w-0 flex-1 flex-col">
        {selectedNode ? (
          <DetailPanel node={selectedNode} neighborData={neighborData} neighborLoading={neighborLoading} neighborError={neighborError} />
        ) : (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-zinc-500">Select a node to explore</p>
          </div>
        )}
      </div>
    </div>
  )
}

function DetailPanel({
  node,
  neighborData,
  neighborLoading,
  neighborError,
}: {
  node: BrowseNodeItem
  neighborData?: { nodes: import('../../api/types').GraphNode[]; edges: import('../../api/types').GraphEdge[] } | null
  neighborLoading: boolean
  neighborError: boolean
}) {
  // Group edges by relationship type
  const edgeGroups = useMemo(() => {
    if (!neighborData?.edges) return []
    const groups: Record<string, { rel: string; targets: string[] }> = {}
    const nodeMap = new Map(neighborData.nodes.map((n) => [n.id, n]))
    for (const edge of neighborData.edges) {
      const rel = edge.relationship
      if (!groups[rel]) groups[rel] = { rel, targets: [] }
      const targetId = edge.src === node.id ? edge.dst : edge.src
      const target = nodeMap.get(targetId)
      groups[rel].targets.push(trimLabel(target?.label ?? targetId, node.namespace))
    }
    return Object.values(groups).sort((a, b) => b.targets.length - a.targets.length)
  }, [neighborData, node.id, node.namespace])

  return (
    <div className="flex h-full flex-col">
      {/* Mini ego graph */}
      <div className="h-64 shrink-0 border-b border-zinc-800">
        {neighborLoading ? (
          <div className="flex h-full items-center justify-center">
            <span className="text-xs text-zinc-500">Loading graph...</span>
          </div>
        ) : neighborData && neighborData.nodes.length > 1 && neighborData.edges.length > 0 ? (
          <SigmaGraph
            nodes={neighborData.nodes}
            edges={neighborData.edges}
            communities={null}
            showCommunities={false}
            showEdges={true}
            focusNodeId={node.id}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <span className="text-xs text-zinc-500">
              {neighborError ? 'Node not in active graph' : 'No neighbor connections'}
            </span>
          </div>
        )}
      </div>

      {/* Node detail */}
      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="mb-1 text-lg font-semibold text-zinc-100 break-words">{trimLabel(node.label, node.namespace)}</h3>
        <div className="mb-4 flex items-center gap-3 text-xs text-zinc-400">
          <span>{node.kind}</span>
          <span>{node.degree} connections</span>
          <span>{(node.centrality * 100).toFixed(1)}% centrality</span>
          {node.namespace && <span>{trimNamespace(node.namespace)}</span>}
        </div>

        {/* Connections grouped by relationship */}
        {edgeGroups.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Connections
            </h4>
            {edgeGroups.map(({ rel, targets }) => {
              const color = EDGE_COLORS[rel] ?? '#52525b60'
              const dotColor = color.length > 7 ? color.slice(0, 7) : color
              return (
                <div key={rel}>
                  <div className="mb-1 flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: dotColor }}
                    />
                    <span className="text-xs font-medium text-zinc-300">{rel}</span>
                    <span className="text-xs text-zinc-500">({targets.length})</span>
                  </div>
                  <div className="ml-4 space-y-0.5">
                    {targets.slice(0, 10).map((t, i) => (
                      <p key={i} className="truncate text-xs text-zinc-400">
                        {t}
                      </p>
                    ))}
                    {targets.length > 10 && (
                      <p className="text-xs text-zinc-500">
                        +{targets.length - 10} more
                      </p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
