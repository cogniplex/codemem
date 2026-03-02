import { useState, useEffect, useCallback } from 'react'
import { Save, RotateCcw } from 'lucide-react'
import { useConfig, useUpdateScoringWeights } from '../../api/hooks'
import type { ScoreBreakdown } from '../../api/types'

const WEIGHT_KEYS: { key: keyof ScoreBreakdown; label: string; color: string }[] = [
  { key: 'vector_similarity', label: 'Vector Similarity', color: 'bg-violet-500' },
  { key: 'graph_strength', label: 'Graph Strength', color: 'bg-cyan-500' },
  { key: 'token_overlap', label: 'Token Overlap', color: 'bg-blue-500' },
  { key: 'temporal', label: 'Temporal', color: 'bg-amber-500' },
  { key: 'tag_matching', label: 'Tag Matching', color: 'bg-emerald-500' },
  { key: 'importance', label: 'Importance', color: 'bg-pink-500' },
  { key: 'confidence', label: 'Confidence', color: 'bg-orange-500' },
  { key: 'recency', label: 'Recency', color: 'bg-teal-500' },
]

const DEFAULT_WEIGHTS: ScoreBreakdown = {
  vector_similarity: 0.25,
  graph_strength: 0.25,
  token_overlap: 0.15,
  temporal: 0.10,
  tag_matching: 0.10,
  importance: 0.05,
  confidence: 0.05,
  recency: 0.05,
}

function normalizeWeights(
  weights: ScoreBreakdown,
  changedKey: keyof ScoreBreakdown,
  newValue: number,
): ScoreBreakdown {
  const result = { ...weights }
  result[changedKey] = newValue

  const otherKeys = WEIGHT_KEYS
    .map((w) => w.key)
    .filter((k) => k !== changedKey)
  const otherSum = otherKeys.reduce((sum, k) => sum + result[k], 0)
  const remaining = 1.0 - newValue

  if (otherSum === 0) {
    const share = remaining / otherKeys.length
    for (const k of otherKeys) {
      result[k] = Math.round(share * 1000) / 1000
    }
  } else {
    const scale = remaining / otherSum
    for (const k of otherKeys) {
      result[k] = Math.round(result[k] * scale * 1000) / 1000
    }
  }

  return result
}

export function ScoringWeights() {
  const { data: config } = useConfig()
  const scoringMutation = useUpdateScoringWeights()
  const [weights, setWeights] = useState<ScoreBreakdown>(DEFAULT_WEIGHTS)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (config && typeof config === 'object' && 'scoring' in config) {
      const scoring = config.scoring as Partial<ScoreBreakdown>
      // eslint-disable-next-line react-hooks/set-state-in-effect -- sync weights from config on load
      setWeights((prev) => ({ ...prev, ...scoring }))
    }
  }, [config])

  const handleChange = useCallback(
    (key: keyof ScoreBreakdown, value: number) => {
      setWeights((prev) => normalizeWeights(prev, key, value))
      setSaved(false)
    },
    [],
  )

  const handleSave = () => {
    scoringMutation.mutate(weights, { onSuccess: () => setSaved(true) })
  }

  const handleReset = () => {
    setWeights(DEFAULT_WEIGHTS)
    setSaved(false)
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850">
      <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
        <h3 className="text-sm font-medium text-zinc-300">Scoring Weights</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            <RotateCcw size={12} />
            Reset
          </button>
          <button
            onClick={handleSave}
            disabled={scoringMutation.isPending}
            className="flex items-center gap-1.5 rounded bg-violet-600 px-3 py-1 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
          >
            <Save size={12} />
            {scoringMutation.isPending ? 'Saving...' : saved ? 'Saved' : 'Save'}
          </button>
        </div>
      </div>
      <div className="space-y-4 p-5">
        {WEIGHT_KEYS.map(({ key, label, color }) => (
          <div key={key}>
            <div className="mb-1.5 flex items-center justify-between">
              <label className="text-xs text-zinc-400">{label}</label>
              <span className="text-xs font-medium tabular-nums text-zinc-200">
                {(weights[key] * 100).toFixed(1)}%
              </span>
            </div>
            <div className="relative">
              <div className="pointer-events-none absolute top-1/2 h-2 w-full -translate-y-1/2 rounded-full bg-zinc-800">
                <div
                  className={`h-full rounded-full ${color}`}
                  style={{ width: `${weights[key] * 100}%` }}
                />
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={Math.round(weights[key] * 100)}
                onChange={(e) =>
                  handleChange(key, Number(e.target.value) / 100)
                }
                className="relative z-10 block h-2 w-full cursor-pointer appearance-none bg-transparent [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-zinc-200 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-zinc-200"
              />
            </div>
          </div>
        ))}
        <p className="text-xs text-zinc-600">
          Total:{' '}
          {(
            Object.values(weights).reduce((s, v) => s + v, 0) * 100
          ).toFixed(1)}
          % (weights auto-normalize to 100%)
        </p>
      </div>
    </div>
  )
}
