import { useState } from 'react'
import { Search, ChevronLeft, ChevronRight, Filter } from 'lucide-react'
import { useMemories, useNamespaces, useSearch } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MemoryDetail } from './MemoryDetail'
import type { ScoreBreakdown } from '../../api/types'

const PAGE_SIZE = 20

const memoryTypes = ['decision', 'pattern', 'preference', 'style', 'habit', 'insight', 'context']

const typeColors: Record<string, string> = {
  decision: 'bg-violet-500/20 text-violet-400',
  pattern: 'bg-cyan-500/20 text-cyan-400',
  preference: 'bg-amber-500/20 text-amber-400',
  style: 'bg-pink-500/20 text-pink-400',
  habit: 'bg-emerald-500/20 text-emerald-400',
  insight: 'bg-blue-500/20 text-blue-400',
  context: 'bg-zinc-500/20 text-zinc-400',
}

export function MemoryBrowser() {
  const globalNamespace = useNamespaceStore((s) => s.active)
  const { data: namespaces } = useNamespaces()

  const [typeFilter, setTypeFilter] = useState<string>('')
  const [nsFilter, setNsFilter] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedBreakdown, setSelectedBreakdown] = useState<ScoreBreakdown | undefined>()

  const activeNamespace = nsFilter || globalNamespace || undefined
  const isSearching = searchQuery.trim().length > 0

  // List query (when not searching)
  const { data: listData, isLoading: listLoading } = useMemories({
    namespace: activeNamespace,
    type: typeFilter || undefined,
    offset: page * PAGE_SIZE,
    limit: PAGE_SIZE,
  })

  // Search query (when searching)
  const { data: searchData, isLoading: searchLoading } = useSearch(
    {
      q: searchQuery.trim(),
      namespace: activeNamespace,
      type: typeFilter || undefined,
      k: PAGE_SIZE,
    },
    isSearching,
  )

  const isLoading = isSearching ? searchLoading : listLoading

  // Normalize results for the table
  const rows = isSearching
    ? (searchData?.results ?? []).map((r) => ({
        id: r.id,
        content: r.content,
        memory_type: r.memory_type,
        tags: r.tags,
        score: r.score,
        score_breakdown: r.score_breakdown,
        namespace: r.namespace,
      }))
    : (listData?.memories ?? []).map((m) => ({
        id: m.id,
        content: m.content,
        memory_type: m.memory_type,
        tags: m.tags,
        score: undefined as number | undefined,
        score_breakdown: undefined as ScoreBreakdown | undefined,
        namespace: m.namespace,
      }))

  const total = isSearching ? (searchData?.results.length ?? 0) : (listData?.total ?? 0)
  const totalPages = isSearching ? 1 : Math.max(1, Math.ceil(total / PAGE_SIZE))

  const handleRowClick = (id: string, breakdown?: ScoreBreakdown) => {
    setSelectedId(id)
    setSelectedBreakdown(breakdown)
  }

  return (
    <div className="flex h-full gap-0">
      {/* Main list area */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Filter bar */}
        <div className="flex flex-wrap items-center gap-3 pb-4">
          {/* Search input */}
          <div className="relative flex-1">
            <Search
              size={16}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500"
            />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                setPage(0)
              }}
              placeholder="Search memories..."
              className="w-full rounded-md border border-zinc-800 bg-zinc-900 py-2 pl-9 pr-3 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:border-zinc-700 focus:ring-1 focus:ring-zinc-700"
            />
          </div>

          {/* Type filter */}
          <div className="flex items-center gap-1.5">
            <Filter size={14} className="text-zinc-500" />
            <select
              value={typeFilter}
              onChange={(e) => {
                setTypeFilter(e.target.value)
                setPage(0)
              }}
              className="rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-2 text-sm text-zinc-300 outline-none focus:border-zinc-700"
            >
              <option value="">All types</option>
              {memoryTypes.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          {/* Namespace filter */}
          <select
            value={nsFilter}
            onChange={(e) => {
              setNsFilter(e.target.value)
              setPage(0)
            }}
            className="max-w-[220px] rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-2 text-sm text-zinc-300 outline-none focus:border-zinc-700"
          >
            <option value="">
              {globalNamespace ? `Namespace: ${globalNamespace}` : 'All namespaces'}
            </option>
            {namespaces?.map((ns) => (
              <option key={ns.name} value={ns.name}>
                {ns.name} ({ns.memory_count})
              </option>
            ))}
          </select>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto rounded-lg border border-zinc-800 bg-zinc-850">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wider text-zinc-500">
                <th className="px-4 py-3 font-medium">Type</th>
                <th className="px-4 py-3 font-medium">Content</th>
                <th className="px-4 py-3 font-medium">Tags</th>
                {isSearching && <th className="px-4 py-3 font-medium">Score</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50">
              {isLoading ? (
                Array.from({ length: 8 }).map((_, i) => (
                  <tr key={i}>
                    <td className="px-4 py-3">
                      <span className="inline-block h-5 w-16 animate-pulse rounded bg-zinc-800" />
                    </td>
                    <td className="px-4 py-3">
                      <span className="inline-block h-4 w-full animate-pulse rounded bg-zinc-800" />
                    </td>
                    <td className="px-4 py-3">
                      <span className="inline-block h-4 w-20 animate-pulse rounded bg-zinc-800" />
                    </td>
                    {isSearching && (
                      <td className="px-4 py-3">
                        <span className="inline-block h-4 w-12 animate-pulse rounded bg-zinc-800" />
                      </td>
                    )}
                  </tr>
                ))
              ) : rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={isSearching ? 4 : 3}
                    className="px-4 py-12 text-center text-zinc-500"
                  >
                    {isSearching ? 'No results found' : 'No memories yet'}
                  </td>
                </tr>
              ) : (
                rows.map((row) => (
                  <tr
                    key={row.id}
                    onClick={() => handleRowClick(row.id, row.score_breakdown)}
                    className={`cursor-pointer transition-colors hover:bg-zinc-800/50 ${
                      selectedId === row.id ? 'bg-zinc-800/70' : ''
                    }`}
                  >
                    <td className="px-4 py-3">
                      <span
                        className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                          typeColors[row.memory_type] ?? typeColors.context
                        }`}
                      >
                        {row.memory_type}
                      </span>
                    </td>
                    <td className="max-w-md px-4 py-3">
                      <p className="line-clamp-2 break-words text-zinc-300">{row.content}</p>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {row.tags.slice(0, 3).map((tag) => (
                          <span
                            key={tag}
                            className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400"
                          >
                            {tag}
                          </span>
                        ))}
                        {row.tags.length > 3 && (
                          <span className="text-xs text-zinc-600">
                            +{row.tags.length - 3}
                          </span>
                        )}
                      </div>
                    </td>
                    {isSearching && (
                      <td className="px-4 py-3 tabular-nums text-zinc-400">
                        {row.score?.toFixed(3)}
                      </td>
                    )}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {!isSearching && (
          <div className="flex items-center justify-between pt-3">
            <p className="text-xs text-zinc-500">
              {total} {total === 1 ? 'memory' : 'memories'}
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
              >
                <ChevronLeft size={16} />
              </button>
              <span className="text-xs tabular-nums text-zinc-400">
                {page + 1} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Detail side panel */}
      {selectedId && (
        <div className="w-80 shrink-0 lg:w-96">
          <MemoryDetail
            memoryId={selectedId}
            scoreBreakdown={selectedBreakdown}
            onClose={() => {
              setSelectedId(null)
              setSelectedBreakdown(undefined)
            }}
          />
        </div>
      )}
    </div>
  )
}
