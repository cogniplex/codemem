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
    <header className="flex h-14 items-center gap-4 border-b border-zinc-800 bg-zinc-950 px-6">
      <button
        onClick={() => setSearchOpen(true)}
        className="flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-400 transition-colors hover:border-zinc-600 hover:text-zinc-300"
      >
        <Search size={14} />
        <span>Search memories...</span>
        <kbd className="ml-4 rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-500">
          /
        </kbd>
      </button>

      <div className="ml-auto flex items-center gap-3">
        <HealthIndicator />

        <label className="text-xs text-zinc-500">Namespace</label>
        <select
          value={active ?? ''}
          onChange={(e) => setActive(e.target.value || null)}
          className="max-w-[200px] rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-zinc-300 outline-none focus:border-zinc-500"
        >
          <option value="">All</option>
          {namespaces?.map((ns) => (
            <option key={ns.name} value={ns.name}>
              {ns.name} ({ns.memory_count})
            </option>
          ))}
        </select>
      </div>
    </header>
  )
}
