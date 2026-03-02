import { FolderSearch, Recycle } from 'lucide-react'
import { StatsCards } from './StatsCards'
import { RecentActivity } from './RecentActivity'
import { TypeDistribution } from './TypeDistribution'
import { useRunConsolidation, useIndexRepo, useRepos } from '../../api/hooks'

export function DashboardView() {
  const consolidation = useRunConsolidation()
  const indexRepo = useIndexRepo()
  const { data: repos } = useRepos()

  const handleIndex = () => {
    const first = repos?.[0]
    if (first) indexRepo.mutate(first.id)
  }

  return (
    <div className="space-y-6">
      <StatsCards />

      <div className="grid gap-6 lg:grid-cols-2">
        <RecentActivity />
        <TypeDistribution />
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={handleIndex}
          disabled={indexRepo.isPending || !repos?.length}
          className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-700 disabled:opacity-50 disabled:pointer-events-none"
        >
          <FolderSearch size={16} />
          Index Repo
        </button>
        <button
          onClick={() => consolidation.mutate('decay')}
          disabled={consolidation.isPending}
          className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-700 disabled:opacity-50 disabled:pointer-events-none"
        >
          <Recycle size={16} />
          Run Consolidation
        </button>
      </div>
    </div>
  )
}
