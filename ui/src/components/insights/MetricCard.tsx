import type { LucideIcon } from 'lucide-react'

interface MetricCardProps {
  label: string
  value: number | string
  icon: LucideIcon
  color: string
  isLoading?: boolean
}

export function MetricCard({ label, value, icon: Icon, color, isLoading }: MetricCardProps) {
  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800/50 bg-zinc-900">
      <div className="bg-zinc-800/40 px-4 py-2">
        <span className="text-[11px] font-medium text-zinc-500">{label}</span>
      </div>
      <div className="flex items-center gap-3 px-4 py-3">
        <div className={`rounded-lg bg-zinc-800/60 p-2 ${color}`}>
          <Icon size={15} strokeWidth={1.5} />
        </div>
        <p className="text-2xl font-bold tabular-nums text-zinc-50">
          {isLoading ? (
            <span className="inline-block h-7 w-16 animate-pulse rounded-lg bg-zinc-800/60" />
          ) : (
            typeof value === 'number' ? value.toLocaleString() : value
          )}
        </p>
      </div>
    </div>
  )
}
