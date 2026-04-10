import { useState } from 'react'
import { useHealth } from '../../api/hooks'

export function HealthIndicator() {
  const { data, isLoading } = useHealth()
  const [showDetails, setShowDetails] = useState(false)

  if (isLoading || !data) {
    return <span className="h-2.5 w-2.5 rounded-full bg-zinc-600" />
  }

  const components = [
    { label: 'Storage', status: data.storage?.status },
    { label: 'Vector', status: data.vector?.status },
    { label: 'Graph', status: data.graph?.status },
    { label: 'Embeddings', status: data.embeddings?.status },
  ]

  const allHealthy = components.every(
    (c) => c.status === 'ok' || c.status === 'healthy'
  )
  const anyError = components.some(
    (c) => c.status && c.status !== 'ok' && c.status !== 'healthy' && c.status !== 'degraded'
  )

  const dotColor = anyError
    ? 'bg-red-400'
    : allHealthy
      ? 'bg-emerald-400'
      : 'bg-amber-400'

  return (
    <div className="relative">
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="flex items-center gap-1.5 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300"
      >
        <span className={`h-2 w-2 rounded-full ${dotColor}`} />
        <span className="hidden sm:inline">
          {anyError ? 'Unhealthy' : allHealthy ? 'Healthy' : 'Degraded'}
        </span>
      </button>

      {showDetails && (
        <div className="absolute right-0 top-full z-50 mt-1 w-48 rounded-lg border border-zinc-700 bg-zinc-900 p-3 shadow-xl">
          <p className="mb-2 text-xs font-medium text-zinc-400">System Health</p>
          <div className="space-y-1.5">
            {components.map(({ label, status }) => {
              const isOk = status === 'ok' || status === 'healthy'
              return (
                <div key={label} className="flex items-center justify-between text-xs">
                  <span className="text-zinc-300">{label}</span>
                  <span className={isOk ? 'text-emerald-400' : 'text-amber-400'}>
                    {status ?? 'unknown'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
