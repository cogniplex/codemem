import { FolderGit2, RefreshCw, Clock, Loader2 } from 'lucide-react'
import { useIndexRepo } from '../../api/hooks'
import type { Repository } from '../../api/types'

const statusStyles: Record<string, string> = {
  idle: 'bg-zinc-500/20 text-zinc-400',
  indexing: 'bg-amber-500/20 text-amber-400',
  indexed: 'bg-emerald-500/20 text-emerald-400',
  error: 'bg-red-500/20 text-red-400',
}

export function RepoCard({ repo }: { repo: Repository }) {
  const indexMutation = useIndexRepo()

  const handleReindex = () => {
    indexMutation.mutate(repo.id)
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <div className="shrink-0 rounded-md bg-zinc-800 p-2 text-violet-400">
            <FolderGit2 size={18} />
          </div>
          <div className="min-w-0">
            <h3 className="truncate text-sm font-medium text-zinc-200">
              {repo.name || repo.path.split('/').pop()}
            </h3>
            <p className="mt-0.5 text-xs text-zinc-500 font-mono truncate max-w-full" title={repo.path}>
              {repo.path}
            </p>
          </div>
        </div>
        <span
          className={`flex items-center gap-1.5 rounded px-2 py-0.5 text-xs font-medium ${
            statusStyles[repo.status] ?? statusStyles.idle
          }`}
        >
          {repo.status === 'indexing' && <Loader2 size={12} className="animate-spin" />}
          {repo.status}
        </span>
      </div>

      {/* Details */}
      <div className="mt-4 space-y-2">
        {repo.namespace && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Namespace:</span>
            <span className="max-w-[180px] truncate rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300" title={repo.namespace}>
              {repo.namespace}
            </span>
          </div>
        )}

        {repo.last_indexed_at && (
          <div className="flex items-center gap-1.5 text-xs text-zinc-500">
            <Clock size={12} />
            <span>Indexed {new Date(repo.last_indexed_at).toLocaleString()}</span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="mt-4 border-t border-zinc-800 pt-3">
        <button
          onClick={handleReindex}
          disabled={indexMutation.isPending || repo.status === 'indexing'}
          className="flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-700 disabled:opacity-50"
        >
          <RefreshCw
            size={12}
            className={indexMutation.isPending ? 'animate-spin' : ''}
          />
          {indexMutation.isPending ? 'Indexing...' : 'Re-index'}
        </button>
      </div>
    </div>
  )
}
