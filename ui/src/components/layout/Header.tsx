import { Search } from 'lucide-react'
import { useNamespaces } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { useUiStore } from '../../stores/ui'
import { HealthIndicator } from '../dashboard/HealthIndicator'

export function Header() {
  const { data: namespaces } = useNamespaces()
  const { active, setActive } = useNamespaceStore()
  const { setSearchOpen } = useUiStore()

  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-zinc-800/60 bg-zinc-950 px-5">
      {/* Search */}
      <button
        onClick={() => setSearchOpen(true)}
        className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-1.5 text-[13px] text-zinc-500 transition-colors hover:border-zinc-700 hover:text-zinc-400"
      >
        <Search size={13} />
        <span>Search memories...</span>
        <kbd className="ml-6 rounded border border-zinc-800 bg-zinc-900 px-1.5 py-0.5 text-[10px] text-zinc-600">
          /
        </kbd>
      </button>

      <div className="ml-auto flex items-center gap-4">
        <HealthIndicator />

        {/* Namespace */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-600">
            Scope
          </span>
          <select
            value={active ?? ''}
            onChange={(e) => setActive(e.target.value || null)}
            className="max-w-[180px] rounded-lg border border-zinc-800 bg-zinc-900/50 px-2.5 py-1 text-[13px] text-zinc-300 outline-none transition-colors hover:border-zinc-700 focus:border-zinc-600"
          >
            <option value="">All namespaces</option>
            {namespaces?.map((ns) => (
              <option key={ns.name} value={ns.name}>
                {ns.name} ({ns.memory_count})
              </option>
            ))}
          </select>
        </div>
      </div>
    </header>
  )
}
