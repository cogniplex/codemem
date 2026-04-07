import { useState, useRef, useEffect } from 'react'
import {
  LayoutDashboard,
  Network,
  Brain,
  Lightbulb,
  Clock,
  Search,
  ChevronDown,
  Check,
} from 'lucide-react'
import { useUiStore } from '../../stores/ui'
import { useNamespaceStore } from '../../stores/namespace'
import { useNamespaces } from '../../api/hooks'
import { HealthIndicator } from '../dashboard/HealthIndicator'

const navItems = [
  { key: 'dashboard' as const, label: 'Dashboard', icon: LayoutDashboard },
  { key: 'graph' as const, label: 'Graph', icon: Network },
  { key: 'memories' as const, label: 'Memories', icon: Brain },
  { key: 'insights' as const, label: 'Insights', icon: Lightbulb },
  { key: 'temporal' as const, label: 'Temporal', icon: Clock },
]

export function Navbar() {
  const { activeView, setActiveView, setSearchOpen } = useUiStore()
  const { active: activeNamespace, setActive: setNamespace } = useNamespaceStore()
  const { data: namespaces } = useNamespaces()

  return (
    <nav className="flex h-12 shrink-0 items-center gap-1.5 border-b border-zinc-800/50 bg-zinc-950 px-4">
      {/* Logo */}
      <div className="flex items-center gap-2 pr-4">
        <div className="flex h-5 w-5 items-center justify-center rounded bg-violet-500/15">
          <Brain size={11} className="text-violet-400" />
        </div>
        <span className="text-[13px] font-semibold tracking-wide text-zinc-200">codemem</span>
      </div>

      {/* Nav tabs */}
      <div className="flex items-center gap-0.5">
        {navItems.map(({ key, label, icon: Icon }) => {
          const active = activeView === key
          return (
            <button
              key={key}
              onClick={() => setActiveView(key)}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors ${
                active
                  ? 'bg-zinc-800/80 text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-800/40 hover:text-zinc-300'
              }`}
            >
              <Icon size={13} strokeWidth={active ? 2 : 1.5} />
              {label}
            </button>
          )
        })}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Search */}
      <button
        onClick={() => setSearchOpen(true)}
        className="flex items-center gap-1.5 rounded-md border border-zinc-800/50 bg-zinc-900/40 px-2.5 py-1 text-[12px] text-zinc-500 transition-colors hover:border-zinc-700 hover:text-zinc-400"
      >
        <Search size={12} />
        <span className="hidden sm:inline">Search</span>
        <kbd className="ml-2 rounded border border-zinc-800 bg-zinc-900 px-1 py-0.5 text-[9px] text-zinc-600">
          /
        </kbd>
      </button>

      {/* Namespace dropdown */}
      <NamespaceDropdown
        value={activeNamespace}
        onChange={setNamespace}
        namespaces={namespaces ?? []}
      />

      {/* Health */}
      <HealthIndicator />
    </nav>
  )
}

function NamespaceDropdown({
  value,
  onChange,
  namespaces,
}: {
  value: string | null
  onChange: (v: string | null) => void
  namespaces: { name: string; memory_count: number }[]
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 rounded-md border border-zinc-800/50 bg-zinc-900/40 px-2.5 py-1 text-[12px] text-zinc-400 transition-colors hover:border-zinc-700 hover:text-zinc-300"
      >
        <span className="max-w-[120px] truncate">
          {value ?? 'All namespaces'}
        </span>
        <ChevronDown size={11} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1.5 w-52 rounded-xl border border-zinc-800/60 bg-zinc-900 p-1.5 shadow-2xl">
          <button
            onClick={() => { onChange(null); setOpen(false) }}
            className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-[12px] transition-colors ${
              !value ? 'bg-zinc-800/60 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/40 hover:text-zinc-200'
            }`}
          >
            All namespaces
            {!value && <Check size={12} className="text-violet-400" />}
          </button>
          {namespaces.map((ns) => (
            <button
              key={ns.name}
              onClick={() => { onChange(ns.name); setOpen(false) }}
              className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-[12px] transition-colors ${
                value === ns.name ? 'bg-zinc-800/60 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/40 hover:text-zinc-200'
              }`}
            >
              <span className="truncate">{ns.name}</span>
              <div className="flex items-center gap-2">
                <span className="text-[10px] tabular-nums text-zinc-600">{ns.memory_count}</span>
                {value === ns.name && <Check size={12} className="text-violet-400" />}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
