import { useState } from 'react'
import { Search, ChevronLeft, ChevronRight, Brain } from 'lucide-react'
import { useMemories, useSearch } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MemoryDetail } from './MemoryDetail'
import type { ScoreBreakdown } from '../../api/types'

const PAGE_SIZE = 20

const memoryTypes = ['decision', 'pattern', 'preference', 'style', 'habit', 'insight', 'context']

const typeStyles: Record<string, { pill: string; dot: string }> = {
  decision: { pill: 'bg-violet-500/10 text-violet-400 border-violet-500/20', dot: 'bg-violet-400' },
  pattern: { pill: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20', dot: 'bg-cyan-400' },
  preference: { pill: 'bg-amber-500/10 text-amber-400 border-amber-500/20', dot: 'bg-amber-400' },
  style: { pill: 'bg-pink-500/10 text-pink-400 border-pink-500/20', dot: 'bg-pink-400' },
  habit: { pill: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', dot: 'bg-emerald-400' },
  insight: { pill: 'bg-blue-500/10 text-blue-400 border-blue-500/20', dot: 'bg-blue-400' },
  context: { pill: 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20', dot: 'bg-zinc-500' },
}

export function MemoryBrowser() {
  const activeNamespace = useNamespaceStore((s) => s.active) ?? undefined

  const [typeFilter, setTypeFilter] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedBreakdown, setSelectedBreakdown] = useState<ScoreBreakdown | undefined>()
  const isSearching = searchQuery.trim().length > 0

  const { data: listData, isLoading: listLoading } = useMemories({
    namespace: activeNamespace,
    type: typeFilter || undefined,
    offset: page * PAGE_SIZE,
    limit: PAGE_SIZE,
  })

  const { data: searchData, isLoading: searchLoading } = useSearch(
    { q: searchQuery.trim(), namespace: activeNamespace, type: typeFilter || undefined, k: PAGE_SIZE },
    isSearching,
  )

  const isLoading = isSearching ? searchLoading : listLoading

  const rows = isSearching
    ? (searchData?.results ?? []).map((r) => ({
        id: r.id, content: r.content, memory_type: r.memory_type,
        tags: r.tags, score: r.score, score_breakdown: r.score_breakdown, namespace: r.namespace,
      }))
    : (listData?.memories ?? []).map((m) => ({
        id: m.id, content: m.content, memory_type: m.memory_type,
        tags: m.tags, score: undefined as number | undefined,
        score_breakdown: undefined as ScoreBreakdown | undefined, namespace: m.namespace,
      }))

  const total = isSearching ? (searchData?.results.length ?? 0) : (listData?.total ?? 0)
  const totalPages = isSearching ? 1 : Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="flex h-full gap-0">
      {/* Main list */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Header */}
        <div className="mb-5 flex items-center gap-2">
          <Brain size={20} className="text-zinc-400" />
          <h2 className="text-lg font-semibold text-zinc-100">Memories</h2>
          <span className="ml-2 rounded-full bg-zinc-800/80 px-2.5 py-0.5 text-[11px] tabular-nums text-zinc-500">
            {total}
          </span>
        </div>

        {/* Search bar */}
        <div className="relative mb-4">
          <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-zinc-600" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setPage(0) }}
            placeholder="Search memories by content..."
            className="w-full rounded-xl border border-zinc-800/50 bg-zinc-900 py-2.5 pl-10 pr-4 text-[13px] text-zinc-200 placeholder-zinc-600 outline-none transition-all focus:border-zinc-600 focus:bg-zinc-900/80 focus:ring-1 focus:ring-zinc-700/50"
          />
        </div>

        {/* Filter pills */}
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <button
            onClick={() => { setTypeFilter(''); setPage(0) }}
            className={`rounded-lg border px-3 py-1 text-[12px] font-medium transition-colors ${
              !typeFilter
                ? 'border-violet-500/30 bg-violet-500/10 text-violet-400'
                : 'border-zinc-800/50 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300'
            }`}
          >
            All
          </button>
          {memoryTypes.map((t) => {
            const active = typeFilter === t
            const s = typeStyles[t] ?? typeStyles.context
            return (
              <button
                key={t}
                onClick={() => { setTypeFilter(active ? '' : t); setPage(0) }}
                className={`flex items-center gap-1.5 rounded-lg border px-3 py-1 text-[12px] font-medium capitalize transition-colors ${
                  active ? `${s.pill} border` : 'border-zinc-800/50 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300'
                }`}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${active ? s.dot : 'bg-zinc-600'}`} />
                {t}
              </button>
            )
          })}

        </div>

        {/* Card list */}
        <div className="flex-1 space-y-2 overflow-y-auto pb-2">
          {isLoading ? (
            Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="rounded-xl border border-zinc-800/40 bg-zinc-900/50 p-4">
                <div className="flex items-center gap-3">
                  <span className="h-5 w-14 animate-pulse rounded-md bg-zinc-800/60" />
                  <span className="h-4 w-3/4 animate-pulse rounded-md bg-zinc-800/40" />
                </div>
              </div>
            ))
          ) : rows.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="rounded-full bg-zinc-800/40 p-4">
                <Brain size={24} className="text-zinc-600" />
              </div>
              <p className="mt-4 text-[13px] text-zinc-500">
                {isSearching ? 'No results found' : 'No memories yet'}
              </p>
            </div>
          ) : (
            rows.map((row) => {
              const s = typeStyles[row.memory_type] ?? typeStyles.context
              const isSelected = selectedId === row.id
              return (
                <button
                  key={row.id}
                  onClick={() => { setSelectedId(row.id); setSelectedBreakdown(row.score_breakdown) }}
                  className={`group w-full rounded-xl border p-4 text-left transition-all duration-150 ${
                    isSelected
                      ? 'border-violet-500/30 bg-violet-500/5'
                      : 'border-zinc-800/40 bg-zinc-900/50 hover:border-zinc-700/50 hover:bg-zinc-800/30'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    {/* Type badge */}
                    <span className={`mt-0.5 shrink-0 rounded-md border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${s.pill}`}>
                      {row.memory_type}
                    </span>

                    {/* Content */}
                    <div className="min-w-0 flex-1">
                      <p className="line-clamp-2 text-[13px] leading-relaxed text-zinc-300 group-hover:text-zinc-200">
                        {row.content}
                      </p>

                      {/* Tags */}
                      {row.tags.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {row.tags.slice(0, 4).map((tag) => (
                            <span
                              key={tag}
                              className="rounded-md bg-zinc-800/60 px-1.5 py-0.5 text-[10px] text-zinc-500"
                            >
                              {tag}
                            </span>
                          ))}
                          {row.tags.length > 4 && (
                            <span className="text-[10px] text-zinc-600">+{row.tags.length - 4}</span>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Score */}
                    {row.score !== undefined && (
                      <span className="shrink-0 rounded-md bg-violet-500/10 px-2 py-0.5 text-[11px] font-medium tabular-nums text-violet-400">
                        {row.score.toFixed(3)}
                      </span>
                    )}
                  </div>
                </button>
              )
            })
          )}
        </div>

        {/* Pagination */}
        {!isSearching && rows.length > 0 && (
          <div className="flex items-center justify-between border-t border-zinc-800/40 pt-3">
            <p className="text-[11px] text-zinc-600">
              Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total}
            </p>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30"
              >
                <ChevronLeft size={14} />
              </button>
              <span className="px-2 text-[11px] tabular-nums text-zinc-500">
                {page + 1} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selectedId && (
        <div className="w-80 shrink-0 lg:w-96">
          <MemoryDetail
            memoryId={selectedId}
            scoreBreakdown={selectedBreakdown}
            onClose={() => { setSelectedId(null); setSelectedBreakdown(undefined) }}
          />
        </div>
      )}
    </div>
  )
}
