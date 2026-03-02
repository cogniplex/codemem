import { useState, useCallback, useRef } from 'react'
import { Play, Square } from 'lucide-react'
import { useRecipes, useRepos } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import { runRecipe } from '../../api/sse'
import { AgentOutput, type StepEntry } from './AgentOutput'

export function AgentRunner() {
  const { data: recipes, isLoading: recipesLoading } = useRecipes()
  const { data: repos } = useRepos()
  const namespace = useNamespaceStore((s) => s.active)

  const [selectedRecipe, setSelectedRecipe] = useState('')
  const [selectedRepo, setSelectedRepo] = useState('')
  const [running, setRunning] = useState(false)
  const [steps, setSteps] = useState<StepEntry[]>([])
  const [totalSteps, setTotalSteps] = useState(0)
  const [recipeName, setRecipeName] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const cancelRef = useRef<(() => void) | null>(null)

  const handleRun = useCallback(() => {
    if (!selectedRecipe) return

    setRunning(true)
    setSteps([])
    setDone(false)
    setRecipeName(null)

    const cancel = runRecipe(
      {
        recipe: selectedRecipe,
        repo_id: selectedRepo || undefined,
        namespace: namespace ?? undefined,
      },
      {
        onRecipeStart: (data) => {
          setRecipeName(data.name)
          setTotalSteps(data.total_steps)
        },
        onStepStart: (data) => {
          setSteps((prev) => [
            ...prev,
            {
              step: data.step,
              tool: data.tool,
              description: data.description,
              status: 'running',
            },
          ])
        },
        onStepResult: (data) => {
          setSteps((prev) =>
            prev.map((s) =>
              s.step === data.step
                ? { ...s, status: data.success ? 'success' : 'error', result: data.result }
                : s,
            ),
          )
        },
        onComplete: () => {
          setRunning(false)
          setDone(true)
        },
        onError: (data) => {
          setRunning(false)
          setSteps((prev) => [
            ...prev,
            { step: -1, tool: 'error', description: data.error, status: 'error' },
          ])
        },
      },
    )

    cancelRef.current = cancel
  }, [selectedRecipe, selectedRepo, namespace])

  const handleCancel = useCallback(() => {
    cancelRef.current?.()
    setRunning(false)
  }, [])

  const selected = recipes?.find((r) => r.id === selectedRecipe)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100">Agent Runner</h2>
        <p className="mt-1 text-sm text-zinc-500">
          Run predefined analysis recipes on your repositories
        </p>
      </div>

      {/* Config form */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="mb-1.5 block text-xs font-medium text-zinc-400">Recipe</label>
            <select
              value={selectedRecipe}
              onChange={(e) => setSelectedRecipe(e.target.value)}
              disabled={running}
              className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-zinc-500 disabled:opacity-50"
            >
              <option value="">Select a recipe...</option>
              {recipesLoading ? (
                <option disabled>Loading...</option>
              ) : (
                recipes?.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))
              )}
            </select>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium text-zinc-400">
              Repository (optional)
            </label>
            <select
              value={selectedRepo}
              onChange={(e) => setSelectedRepo(e.target.value)}
              disabled={running}
              className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-zinc-500 disabled:opacity-50"
            >
              <option value="">None</option>
              {repos?.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name ?? r.path}
                </option>
              ))}
            </select>
          </div>
        </div>

        {selected && (
          <div className="mt-3">
            <p className="text-sm text-zinc-400">{selected.description}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {selected.steps.map((s, i) => (
                <span
                  key={i}
                  className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400"
                >
                  {i + 1}. {s.tool}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="mt-4 flex gap-2">
          <button
            onClick={handleRun}
            disabled={!selectedRecipe || running}
            className="flex items-center gap-2 rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50 disabled:hover:bg-violet-600"
          >
            <Play size={14} />
            Run
          </button>
          {running && (
            <button
              onClick={handleCancel}
              className="flex items-center gap-2 rounded-md border border-zinc-700 px-4 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
            >
              <Square size={14} />
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Output */}
      <AgentOutput
        steps={steps}
        totalSteps={totalSteps}
        recipeName={recipeName}
        done={done}
      />
    </div>
  )
}
