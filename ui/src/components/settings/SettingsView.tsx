import {
  Settings,
  Play,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react'
import { useHealth, useConsolidationStatus, useRunConsolidation } from '../../api/hooks'
import { ScoringWeights } from './ScoringWeights'

function HealthBadge({ status }: { status: string }) {
  if (status === 'ok' || status === 'healthy') {
    return (
      <span className="flex items-center gap-1 text-xs text-emerald-400">
        <CheckCircle size={12} />
        Healthy
      </span>
    )
  }
  if (status === 'degraded') {
    return (
      <span className="flex items-center gap-1 text-xs text-amber-400">
        <AlertCircle size={12} />
        Degraded
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1 text-xs text-red-400">
      <XCircle size={12} />
      {status}
    </span>
  )
}

function ConsolidationSection() {
  const { data, isLoading } = useConsolidationStatus()
  const runConsolidation = useRunConsolidation()

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850">
      <div className="border-b border-zinc-800 px-5 py-3">
        <h3 className="text-sm font-medium text-zinc-300">
          Consolidation Cycles
        </h3>
      </div>
      <div className="divide-y divide-zinc-800/50">
        {isLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3 px-5 py-3">
              <span className="h-4 w-20 animate-pulse rounded bg-zinc-800" />
              <span className="ml-auto h-4 w-28 animate-pulse rounded bg-zinc-800" />
            </div>
          ))
        ) : !data?.cycles.length ? (
          <p className="px-5 py-6 text-center text-sm text-zinc-500">
            No consolidation data
          </p>
        ) : (
          data.cycles.map((c) => (
            <div
              key={c.cycle}
              className="flex items-center gap-4 px-5 py-3"
            >
              <span className="text-sm font-medium capitalize text-zinc-200">
                {c.cycle}
              </span>
              <span className="text-xs text-zinc-500">
                {c.last_run
                  ? `Last run: ${new Date(c.last_run).toLocaleDateString()}`
                  : 'Never run'}
              </span>
              <span className="text-xs text-zinc-600">
                {c.affected_count} affected
              </span>
              <button
                onClick={() => runConsolidation.mutate(c.cycle)}
                disabled={runConsolidation.isPending}
                className="ml-auto flex items-center gap-1.5 rounded bg-zinc-800 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
              >
                {runConsolidation.isPending ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : (
                  <Play size={12} />
                )}
                Run
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

function HealthSection() {
  const { data, isLoading } = useHealth()

  const components = data
    ? [
        { label: 'Storage', health: data.storage },
        { label: 'Vector Index', health: data.vector },
        { label: 'Graph Engine', health: data.graph },
        { label: 'Embeddings', health: data.embeddings },
      ]
    : []

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850">
      <div className="border-b border-zinc-800 px-5 py-3">
        <h3 className="text-sm font-medium text-zinc-300">System Health</h3>
      </div>
      <div className="grid grid-cols-2 gap-4 p-5 lg:grid-cols-4">
        {isLoading
          ? Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="rounded-md border border-zinc-800 bg-zinc-900 p-4"
              >
                <span className="block h-4 w-20 animate-pulse rounded bg-zinc-800" />
                <span className="mt-2 block h-3 w-14 animate-pulse rounded bg-zinc-800" />
              </div>
            ))
          : components.map(({ label, health }) => (
              <div
                key={label}
                className="rounded-md border border-zinc-800 bg-zinc-900 p-4"
              >
                <p className="text-sm font-medium text-zinc-300">{label}</p>
                <div className="mt-1">
                  <HealthBadge status={health.status} />
                </div>
                {health.detail && (
                  <p className="mt-1 text-xs text-zinc-600">{health.detail}</p>
                )}
              </div>
            ))}
      </div>
    </div>
  )
}

export function SettingsView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Settings size={20} className="text-zinc-400" />
        <h2 className="text-lg font-semibold text-zinc-100">Settings</h2>
      </div>

      <ScoringWeights />
      <ConsolidationSection />
      <HealthSection />
    </div>
  )
}
