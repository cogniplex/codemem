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
    <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-5">
      <div className="flex items-center gap-3">
        <div className={`rounded-md bg-zinc-800 p-2 ${color}`}>
          <Icon size={18} />
        </div>
        <span className="text-sm text-zinc-400">{label}</span>
      </div>
      <p className="mt-3 text-2xl font-semibold tabular-nums text-zinc-100">
        {isLoading ? (
          <span className="inline-block h-7 w-16 animate-pulse rounded bg-zinc-800" />
        ) : (
          typeof value === 'number' ? value.toLocaleString() : value
        )}
      </p>
    </div>
  )
}
