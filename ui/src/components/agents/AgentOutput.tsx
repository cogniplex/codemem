import { CheckCircle, XCircle, Loader2 } from 'lucide-react'

export interface StepEntry {
  step: number
  tool: string
  description: string
  status: 'running' | 'success' | 'error'
  result?: string
}

interface AgentOutputProps {
  steps: StepEntry[]
  totalSteps: number
  recipeName: string | null
  done: boolean
}

export function AgentOutput({ steps, totalSteps, recipeName, done }: AgentOutputProps) {
  if (!recipeName) return null

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950">
      <div className="flex items-center gap-3 border-b border-zinc-800 px-5 py-3">
        <h3 className="text-sm font-medium text-zinc-300">{recipeName}</h3>
        <span className="text-xs text-zinc-500">
          {steps.filter((s) => s.status !== 'running').length}/{totalSteps} steps
        </span>
        {done && (
          <span className="ml-auto rounded bg-emerald-500/20 px-2 py-0.5 text-xs font-medium text-emerald-400">
            Complete
          </span>
        )}
      </div>
      <div className="divide-y divide-zinc-800/50">
        {steps.map((entry) => (
          <div key={entry.step} className="px-5 py-3">
            <div className="flex items-center gap-2">
              {entry.status === 'running' ? (
                <Loader2 size={14} className="animate-spin text-blue-400" />
              ) : entry.status === 'success' ? (
                <CheckCircle size={14} className="text-emerald-400" />
              ) : (
                <XCircle size={14} className="text-red-400" />
              )}
              <span className="shrink-0 text-sm font-medium text-zinc-200">{entry.tool}</span>
              <span className="min-w-0 truncate text-xs text-zinc-500">{entry.description}</span>
            </div>
            {entry.result && (
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-zinc-900 p-3 text-xs text-zinc-400">
                {entry.result}
              </pre>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
