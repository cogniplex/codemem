import { useState } from 'react'
import { Plus, FolderGit2, X } from 'lucide-react'
import { useRepos, useRegisterRepo } from '../../api/hooks'
import { RepoCard } from './RepoCard'
import { IndexProgress } from './IndexProgress'

export function RepoManager() {
  const { data: repos, isLoading } = useRepos()
  const registerMutation = useRegisterRepo()

  const [showForm, setShowForm] = useState(false)
  const [newPath, setNewPath] = useState('')
  const [newName, setNewName] = useState('')

  const handleAdd = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newPath.trim()) return
    registerMutation.mutate(
      { path: newPath.trim(), name: newName.trim() || undefined },
      {
        onSuccess: () => {
          setNewPath('')
          setNewName('')
          setShowForm(false)
        },
      },
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-zinc-100">Repositories</h2>
          <p className="mt-1 text-sm text-zinc-500">
            Manage indexed code repositories
          </p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 rounded-md bg-violet-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500"
        >
          {showForm ? <X size={16} /> : <Plus size={16} />}
          {showForm ? 'Cancel' : 'Add Repository'}
        </button>
      </div>

      {/* Add form */}
      {showForm && (
        <form
          onSubmit={handleAdd}
          className="rounded-lg border border-zinc-800 bg-zinc-850 p-5"
        >
          <h3 className="mb-4 text-sm font-medium text-zinc-300">
            Register a new repository
          </h3>
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-zinc-500">
                Repository path *
              </label>
              <input
                type="text"
                value={newPath}
                onChange={(e) => setNewPath(e.target.value)}
                placeholder="/path/to/repository"
                required
                className="w-full rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:border-zinc-700 focus:ring-1 focus:ring-zinc-700"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-zinc-500">
                Display name (optional)
              </label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="my-project"
                className="w-full rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:border-zinc-700 focus:ring-1 focus:ring-zinc-700"
              />
            </div>
            <button
              type="submit"
              disabled={registerMutation.isPending || !newPath.trim()}
              className="flex items-center gap-2 rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
            >
              <FolderGit2 size={14} />
              {registerMutation.isPending ? 'Registering...' : 'Register'}
            </button>
            {registerMutation.isError && (
              <p className="text-xs text-red-400">
                {(registerMutation.error as Error).message}
              </p>
            )}
          </div>
        </form>
      )}

      {/* Index progress (SSE) */}
      <IndexProgress />

      {/* Repo grid */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="h-40 animate-pulse rounded-lg border border-zinc-800 bg-zinc-850"
            />
          ))}
        </div>
      ) : !repos || repos.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-zinc-800 py-16">
          <FolderGit2 size={32} className="mb-3 text-zinc-600" />
          <p className="text-sm text-zinc-500">No repositories registered</p>
          <p className="mt-1 text-xs text-zinc-600">
            Add a repository to start indexing code
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {repos.map((repo) => (
            <RepoCard key={repo.id} repo={repo} />
          ))}
        </div>
      )}
    </div>
  )
}
