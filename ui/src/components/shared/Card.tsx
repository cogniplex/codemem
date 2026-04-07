import type { ReactNode } from 'react'

interface CardProps {
  title?: string
  icon?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  padded?: boolean
}

/**
 * Shared card component.
 * Header: bg-zinc-900 (darker). Body: bg-zinc-850 (lighter).
 */
export function Card({ title, icon, actions, children, className = '', padded = true }: CardProps) {
  return (
    <div className={`flex flex-col overflow-hidden rounded-xl border border-zinc-800/40 bg-zinc-850 ${className}`}>
      {title && (
        <div className="flex shrink-0 items-center justify-between bg-zinc-900 px-5 py-2">
          <div className="flex items-center gap-2">
            {icon}
            <h3 className="text-[14px] font-medium text-white">{title}</h3>
          </div>
          {actions}
        </div>
      )}
      <div className={`min-h-0 flex-1 ${padded ? 'px-5 py-4' : ''}`}>{children}</div>
    </div>
  )
}

/** Standard metric card for insight grids — header has label, body has icon + value */
export function MetricCard({
  label,
  value,
  icon,
  isLoading,
}: {
  label: string
  value: number | string
  icon?: ReactNode
  isLoading?: boolean
}) {
  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-zinc-800/40 bg-zinc-850">
      <div className="flex shrink-0 items-center gap-2 bg-zinc-900 px-4 py-1.5">
        {icon}
        <span className="text-[13px] font-medium text-zinc-200">{label}</span>
      </div>
      <div className="flex-1 px-4 py-3">
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

/** Compact metric card for tight spaces (e.g. memory detail sidebar) */
export function MiniMetricCard({
  label,
  value,
  icon,
}: {
  label: string
  value: string | number
  icon?: ReactNode
}) {
  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-zinc-800/40 bg-zinc-850">
      <div className="flex shrink-0 items-center gap-1.5 bg-zinc-900 px-3 py-1">
        {icon}
        <span className="text-[11px] font-medium text-zinc-300">{label}</span>
      </div>
      <div className="flex-1 px-3 py-2 text-center">
        <p className="text-base font-semibold tabular-nums text-zinc-100">
          {typeof value === 'number' ? value.toLocaleString() : value}
        </p>
      </div>
    </div>
  )
}
