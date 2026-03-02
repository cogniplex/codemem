import { ShieldAlert, Globe, Lock } from 'lucide-react'
import { useSecurityInsights } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { MetricCard } from './MetricCard'
import type { MemoryItem } from '../../api/types'

function severityColor(tags: string[]) {
  if (tags.some((t) => t.includes('severity:high'))) return 'border-red-500/30 bg-red-500/5'
  if (tags.some((t) => t.includes('severity:medium')))
    return 'border-amber-500/30 bg-amber-500/5'
  return 'border-zinc-700 bg-zinc-900/50'
}

function severityBadge(tags: string[]) {
  if (tags.some((t) => t.includes('severity:high')))
    return { text: 'high', cls: 'bg-red-500/15 text-red-400' }
  if (tags.some((t) => t.includes('severity:medium')))
    return { text: 'medium', cls: 'bg-amber-500/15 text-amber-400' }
  return { text: 'info', cls: 'bg-zinc-700/50 text-zinc-400' }
}

function SecurityFeed({ insights }: { insights: MemoryItem[] }) {
  if (insights.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-zinc-500">
        No security insights yet. Run the enrichment pipeline to scan for security patterns.
      </p>
    )
  }

  return (
    <div className="space-y-2">
      {insights.map((item) => {
        const badge = severityBadge(item.tags)
        return (
          <div
            key={item.id}
            className={`rounded-md border px-4 py-3 ${severityColor(item.tags)}`}
          >
            <div className="flex items-start justify-between gap-3">
              <p className="text-sm text-zinc-300">{item.content}</p>
              <span
                className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${badge.cls}`}
              >
                {badge.text}
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-600">
              {new Date(item.created_at).toLocaleDateString()}
            </p>
          </div>
        )
      })}
    </div>
  )
}

export function SecurityTab() {
  const namespace = useNamespaceStore((s) => s.activeNamespace)
  const { data, isLoading } = useSecurityInsights(
    namespace ? { namespace } : undefined,
  )

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="Sensitive Files"
          value={data?.sensitive_file_count ?? 0}
          icon={ShieldAlert}
          color="text-red-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="API Endpoints"
          value={data?.endpoint_count ?? 0}
          icon={Globe}
          color="text-amber-400"
          isLoading={isLoading}
        />
        <MetricCard
          label="Security Functions"
          value={data?.security_function_count ?? 0}
          icon={Lock}
          color="text-emerald-400"
          isLoading={isLoading}
        />
      </div>

      <div>
        <h3 className="mb-3 text-sm font-medium text-zinc-300">Security Findings</h3>
        <SecurityFeed insights={data?.insights ?? []} />
      </div>
    </div>
  )
}
