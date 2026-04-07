import { ShieldAlert, Globe, Lock } from 'lucide-react'
import { useSecurityInsights } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { Card, MetricCard } from '../shared/Card'
import type { MemoryItem } from '../../api/types'

function severityBadge(tags: string[]) {
  if (tags.some((t) => t.includes('severity:high')))
    return { text: 'high', cls: 'bg-red-500/10 text-red-400' }
  if (tags.some((t) => t.includes('severity:medium')))
    return { text: 'medium', cls: 'bg-amber-500/10 text-amber-400' }
  return { text: 'info', cls: 'bg-zinc-700/30 text-zinc-400' }
}

function SecurityFeed({ insights }: { insights: MemoryItem[] }) {
  if (insights.length === 0) {
    return <p className="py-6 text-center text-[13px] text-zinc-600">No security findings yet.</p>
  }

  return (
    <div className="space-y-2">
      {insights.map((item) => {
        const badge = severityBadge(item.tags)
        return (
          <div key={item.id} className="rounded-lg border border-zinc-700/30 bg-zinc-900 px-4 py-3 transition-colors hover:bg-zinc-800/80">
            <div className="flex items-start justify-between gap-3">
              <p className="min-w-0 break-words text-[13px] leading-relaxed text-zinc-300">{item.content}</p>
              <span className={`shrink-0 rounded-md px-2 py-0.5 text-[11px] font-medium ${badge.cls}`}>
                {badge.text}
              </span>
            </div>
            <p className="mt-1 text-[11px] text-zinc-600">{new Date(item.created_at).toLocaleDateString()}</p>
          </div>
        )
      })}
    </div>
  )
}

export function SecurityTab() {
  const namespace = useNamespaceStore((s) => s.active)
  const { data, isLoading } = useSecurityInsights(namespace ? { namespace } : undefined)

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Sensitive Files" value={data?.sensitive_file_count ?? 0} icon={<ShieldAlert size={15} className="text-red-400" />} isLoading={isLoading} />
        <MetricCard label="API Endpoints" value={data?.endpoint_count ?? 0} icon={<Globe size={15} className="text-amber-400" />} isLoading={isLoading} />
        <MetricCard label="Security Functions" value={data?.security_function_count ?? 0} icon={<Lock size={15} className="text-cyan-400" />} isLoading={isLoading} />
      </div>

      <Card title="Security Findings">
        <SecurityFeed insights={data?.insights ?? []} />
      </Card>
    </div>
  )
}
