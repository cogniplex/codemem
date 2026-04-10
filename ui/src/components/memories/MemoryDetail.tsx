import { X, Trash2, Clock, Eye, Star } from 'lucide-react'
import { useMemory, useDeleteMemory } from '../../api/hooks'
import { ScoreRadar } from './ScoreRadar'
import { getTypeColors } from '../../utils/colors'
import { MiniMetricCard } from '../shared/Card'
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
      <div className="flex h-full items-center justify-center bg-zinc-950 p-6">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-700 border-t-violet-400" />
      </div>
    )
  }

  if (!memory) {
    return (
      <div className="flex h-full items-center justify-center bg-zinc-950 p-6">
        <p className="text-[13px] text-zinc-500">Memory not found</p>
      </div>
    )
  }

  const tc = getTypeColors(memory.memory_type)

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between bg-zinc-900 px-5 py-2">
        <h3 className="text-[14px] font-medium text-white">Detail</h3>
        <button onClick={onClose} className="rounded-md p-1 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-400">
          <X size={14} />
        </button>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {/* Type + namespace */}
        <div className="flex items-center gap-2">
          <span className={`rounded-md border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${tc.bg} ${tc.text} ${tc.border}`}>
            {memory.memory_type}
          </span>
          {memory.namespace && (
            <span className="max-w-[180px] truncate rounded-md bg-zinc-800/40 px-2 py-0.5 text-[11px] text-zinc-500" title={memory.namespace}>
              {memory.namespace}
            </span>
          )}
        </div>

        {/* Content */}
        <div className="rounded-xl border border-zinc-800/30 bg-zinc-850 p-4">
          <p className="whitespace-pre-wrap break-words text-[13px] leading-relaxed text-zinc-300">
            {memory.content}
          </p>
        </div>

        {/* Metrics — using shared MetricCard */}
        <div className="grid grid-cols-3 gap-2">
          <MiniMetricCard label="Importance" value={memory.importance.toFixed(2)} icon={<Star size={11} className="text-amber-400" />} />
          <MiniMetricCard label="Accesses" value={memory.access_count} icon={<Eye size={11} className="text-cyan-400" />} />
          <MiniMetricCard label="Confidence" value={memory.confidence.toFixed(2)} icon={<Clock size={11} className="text-emerald-400" />} />
        </div>

        {/* Tags */}
        {memory.tags.length > 0 && (
          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">Tags</p>
            <div className="flex flex-wrap gap-1.5">
              {memory.tags.map((tag) => (
                <span key={tag} className="rounded-md bg-zinc-800/40 px-2 py-0.5 text-[11px] text-zinc-400">{tag}</span>
              ))}
            </div>
          </div>
        )}

        {/* Score radar */}
        {scoreBreakdown && (
          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">Score</p>
            <div className="rounded-xl border border-zinc-800/30 bg-zinc-850 p-3">
              <ScoreRadar breakdown={scoreBreakdown} />
            </div>
          </div>
        )}

        {/* Meta */}
        <div className="space-y-1">
          <p className="text-[11px] text-zinc-600">Created {new Date(memory.created_at).toLocaleString()}</p>
          <p className="text-[11px] text-zinc-600">Updated {new Date(memory.updated_at).toLocaleString()}</p>
          <p className="break-all font-mono text-[10px] text-zinc-700">{memory.id}</p>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-zinc-800/30 p-4">
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
