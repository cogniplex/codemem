import { X, Trash2, Clock, Eye, Star } from 'lucide-react'
import { useMemory, useDeleteMemory } from '../../api/hooks'
import { ScoreRadar } from './ScoreRadar'
import { getTypeColors } from '../../utils/colors'
import type { ScoreBreakdown } from '../../api/types'

interface MemoryDetailProps {
  memoryId: string
  scoreBreakdown?: ScoreBreakdown
  onClose: () => void
}

export function MemoryDetail({ memoryId, scoreBreakdown, onClose }: MemoryDetailProps) {
  const { data: memory, isLoading } = useMemory(memoryId)
  const deleteMutation = useDeleteMemory()

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center border-l border-zinc-800/40 bg-zinc-950 p-6">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-700 border-t-violet-400" />
      </div>
    )
  }

  if (!memory) {
    return (
      <div className="flex h-full items-center justify-center border-l border-zinc-800/40 bg-zinc-950 p-6">
        <p className="text-[13px] text-zinc-500">Memory not found</p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col border-l border-zinc-800/40 bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800/40 px-5 py-3.5">
        <h3 className="text-[13px] font-medium text-zinc-100">Detail</h3>
        <button
          onClick={onClose}
          className="rounded-md p-1 text-zinc-600 transition-colors hover:bg-zinc-800 hover:text-zinc-400"
        >
          <X size={14} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 space-y-5 overflow-y-auto p-5">
        {/* Type + namespace */}
        <div className="flex items-center gap-2">
          <span
            className={`rounded-md border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
              (() => { const c = getTypeColors(memory.memory_type); return `${c.bg} ${c.text} ${c.border}` })()
            }`}
          >
            {memory.memory_type}
          </span>
          {memory.namespace && (
            <span className="max-w-[180px] truncate rounded-md bg-zinc-800/50 px-2 py-0.5 text-[11px] text-zinc-500" title={memory.namespace}>
              {memory.namespace}
            </span>
          )}
        </div>

        {/* Content */}
        <div className="rounded-xl border border-zinc-800/40 bg-zinc-900/50 p-4">
          <p className="whitespace-pre-wrap break-words text-[13px] leading-relaxed text-zinc-300">
            {memory.content}
          </p>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-3 gap-2">
          <MetricCard icon={<Star size={13} />} iconColor="text-amber-400" label="Importance" value={memory.importance.toFixed(2)} />
          <MetricCard icon={<Eye size={13} />} iconColor="text-cyan-400" label="Accesses" value={String(memory.access_count)} />
          <MetricCard icon={<Clock size={13} />} iconColor="text-emerald-400" label="Confidence" value={memory.confidence.toFixed(2)} />
        </div>

        {/* Tags */}
        {memory.tags.length > 0 && (
          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">Tags</p>
            <div className="flex flex-wrap gap-1.5">
              {memory.tags.map((tag) => (
                <span key={tag} className="rounded-md bg-zinc-800/60 px-2 py-0.5 text-[11px] text-zinc-400">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Score radar */}
        {scoreBreakdown && (
          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">Score</p>
            <div className="rounded-xl border border-zinc-800/40 bg-zinc-900/50 p-3">
              <ScoreRadar breakdown={scoreBreakdown} />
            </div>
          </div>
        )}

        {/* Meta */}
        <div className="space-y-1">
          <p className="text-[11px] text-zinc-600">
            Created {new Date(memory.created_at).toLocaleString()}
          </p>
          <p className="text-[11px] text-zinc-600">
            Updated {new Date(memory.updated_at).toLocaleString()}
          </p>
          <p className="break-all font-mono text-[10px] text-zinc-700">{memory.id}</p>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-zinc-800/40 p-4">
        <button
          onClick={() => deleteMutation.mutate(memoryId, { onSuccess: onClose })}
          disabled={deleteMutation.isPending}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-red-500/8 px-3 py-2 text-[13px] font-medium text-red-400 transition-colors hover:bg-red-500/15 disabled:opacity-50"
        >
          <Trash2 size={13} />
          {deleteMutation.isPending ? 'Deleting...' : 'Delete Memory'}
        </button>
      </div>
    </div>
  )
}

function MetricCard({ icon, iconColor, label, value }: { icon: React.ReactNode; iconColor: string; label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-800/40 bg-zinc-900/50 p-3 text-center">
      <span className={`mx-auto mb-1 block ${iconColor}`}>{icon}</span>
      <p className="text-base font-semibold tabular-nums text-zinc-100">{value}</p>
      <p className="text-[10px] text-zinc-600">{label}</p>
    </div>
  )
}
