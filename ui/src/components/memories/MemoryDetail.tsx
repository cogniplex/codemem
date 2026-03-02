import { X, Trash2, Clock, Eye, Star } from 'lucide-react'
import { useMemory, useDeleteMemory } from '../../api/hooks'
import { ScoreRadar } from './ScoreRadar'
import type { ScoreBreakdown } from '../../api/types'

const typeColors: Record<string, string> = {
  decision: 'bg-violet-500/20 text-violet-400',
  pattern: 'bg-cyan-500/20 text-cyan-400',
  preference: 'bg-amber-500/20 text-amber-400',
  style: 'bg-pink-500/20 text-pink-400',
  habit: 'bg-emerald-500/20 text-emerald-400',
  insight: 'bg-blue-500/20 text-blue-400',
  context: 'bg-zinc-500/20 text-zinc-400',
}

interface MemoryDetailProps {
  memoryId: string
  scoreBreakdown?: ScoreBreakdown
  onClose: () => void
}

export function MemoryDetail({ memoryId, scoreBreakdown, onClose }: MemoryDetailProps) {
  const { data: memory, isLoading } = useMemory(memoryId)
  const deleteMutation = useDeleteMemory()

  const handleDelete = () => {
    deleteMutation.mutate(memoryId, { onSuccess: onClose })
  }

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center border-l border-zinc-800 bg-zinc-950 p-6">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-600 border-t-violet-400" />
      </div>
    )
  }

  if (!memory) {
    return (
      <div className="flex h-full items-center justify-center border-l border-zinc-800 bg-zinc-950 p-6">
        <p className="text-sm text-zinc-500">Memory not found</p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col border-l border-zinc-800 bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
        <h3 className="text-sm font-medium text-zinc-300">Memory Detail</h3>
        <button
          onClick={onClose}
          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          <X size={16} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 space-y-5 overflow-y-auto p-5">
        {/* Type badge */}
        <div className="flex items-center gap-2">
          <span
            className={`rounded px-2 py-0.5 text-xs font-medium ${
              typeColors[memory.memory_type] ?? typeColors.context
            }`}
          >
            {memory.memory_type}
          </span>
          {memory.namespace && (
            <span className="max-w-[200px] truncate rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400" title={memory.namespace}>
              {memory.namespace}
            </span>
          )}
        </div>

        {/* Full content */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-300">
            {memory.content}
          </p>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-center">
            <Star size={14} className="mx-auto mb-1 text-amber-400" />
            <p className="text-lg font-semibold text-zinc-100">
              {memory.importance.toFixed(2)}
            </p>
            <p className="text-xs text-zinc-500">Importance</p>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-center">
            <Eye size={14} className="mx-auto mb-1 text-cyan-400" />
            <p className="text-lg font-semibold text-zinc-100">{memory.access_count}</p>
            <p className="text-xs text-zinc-500">Accesses</p>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-center">
            <Clock size={14} className="mx-auto mb-1 text-emerald-400" />
            <p className="text-lg font-semibold text-zinc-100">
              {memory.confidence.toFixed(2)}
            </p>
            <p className="text-xs text-zinc-500">Confidence</p>
          </div>
        </div>

        {/* Tags */}
        {memory.tags.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
              Tags
            </p>
            <div className="flex flex-wrap gap-1.5">
              {memory.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full bg-zinc-800 px-2.5 py-0.5 text-xs text-zinc-300"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Score radar */}
        {scoreBreakdown && (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
              Score Breakdown
            </p>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3">
              <ScoreRadar breakdown={scoreBreakdown} />
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="space-y-1 text-xs text-zinc-500">
          <p>Created: {new Date(memory.created_at).toLocaleString()}</p>
          <p>Updated: {new Date(memory.updated_at).toLocaleString()}</p>
          <p className="break-all font-mono text-zinc-600">ID: {memory.id}</p>
        </div>
      </div>

      {/* Footer actions */}
      <div className="border-t border-zinc-800 p-4">
        <button
          onClick={handleDelete}
          disabled={deleteMutation.isPending}
          className="flex w-full items-center justify-center gap-2 rounded-md bg-red-500/10 px-3 py-2 text-sm text-red-400 transition-colors hover:bg-red-500/20 disabled:opacity-50"
        >
          <Trash2 size={14} />
          {deleteMutation.isPending ? 'Deleting...' : 'Delete Memory'}
        </button>
      </div>
    </div>
  )
}
