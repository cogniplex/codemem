import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, X } from 'lucide-react'
import { useSearch } from '../../api/hooks'
import { useUiStore } from '../../stores/ui'
import { useNamespaceStore } from '../../stores/namespace'

const typeColors: Record<string, string> = {
  decision: 'bg-violet-500/20 text-violet-400',
  pattern: 'bg-cyan-500/20 text-cyan-400',
  preference: 'bg-amber-500/20 text-amber-400',
  style: 'bg-pink-500/20 text-pink-400',
  habit: 'bg-emerald-500/20 text-emerald-400',
  insight: 'bg-blue-500/20 text-blue-400',
  context: 'bg-zinc-500/20 text-zinc-400',
}

export function SearchModal() {
  const { searchOpen, setSearchOpen, setActiveView } = useUiStore()
  const namespace = useNamespaceStore((s) => s.active)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 300)
    return () => clearTimeout(timer)
  }, [query])

  // Focus input when modal opens
  useEffect(() => {
    if (searchOpen) {
      setTimeout(() => inputRef.current?.focus(), 50)
    } else {
      setQuery('')
      setDebouncedQuery('')
    }
  }, [searchOpen])

  // Global keyboard shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '/' && !searchOpen && !['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault()
        setSearchOpen(true)
      }
      if (e.key === 'Escape' && searchOpen) {
        setSearchOpen(false)
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [searchOpen, setSearchOpen])

  const isSearching = debouncedQuery.trim().length > 1
  const { data, isLoading } = useSearch(
    { q: debouncedQuery.trim(), namespace: namespace ?? undefined, k: 10 },
    isSearching,
  )

  const handleSelect = useCallback(() => {
    setSearchOpen(false)
    setActiveView('memories')
  }, [setSearchOpen, setActiveView])

  if (!searchOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={() => setSearchOpen(false)}
      />

      {/* Modal */}
      <div className="relative w-full max-w-lg rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl">
        {/* Search input */}
        <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
          <Search size={18} className="shrink-0 text-zinc-500" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search memories..."
            className="flex-1 bg-transparent text-sm text-zinc-200 placeholder-zinc-500 outline-none"
          />
          <button
            onClick={() => setSearchOpen(false)}
            className="shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <X size={16} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[50vh] overflow-y-auto">
          {!isSearching && (
            <p className="px-4 py-6 text-center text-sm text-zinc-500">
              Type to search memories...
            </p>
          )}

          {isSearching && isLoading && (
            <div className="flex items-center justify-center py-6">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-600 border-t-violet-400" />
            </div>
          )}

          {isSearching && !isLoading && data?.results.length === 0 && (
            <p className="px-4 py-6 text-center text-sm text-zinc-500">
              No results found
            </p>
          )}

          {data?.results.map((r) => (
            <button
              key={r.id}
              onClick={handleSelect}
              className="flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-zinc-800/50"
            >
              <span
                className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-xs font-medium ${
                  typeColors[r.memory_type] ?? typeColors.context
                }`}
              >
                {r.memory_type}
              </span>
              <div className="min-w-0 flex-1">
                <p className="line-clamp-2 break-words text-sm text-zinc-300">{r.content}</p>
                <div className="mt-1 flex items-center gap-2">
                  <span className="text-xs tabular-nums text-zinc-500">
                    score: {r.score.toFixed(3)}
                  </span>
                  {r.tags.slice(0, 2).map((tag) => (
                    <span key={tag} className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-500">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Footer hint */}
        <div className="border-t border-zinc-800 px-4 py-2">
          <p className="text-xs text-zinc-600">
            <kbd className="rounded border border-zinc-700 bg-zinc-800 px-1 py-0.5 text-zinc-500">Esc</kbd>
            {' '}to close
            {' '}<kbd className="rounded border border-zinc-700 bg-zinc-800 px-1 py-0.5 text-zinc-500">/</kbd>
            {' '}to open
          </p>
        </div>
      </div>
    </div>
  )
}
