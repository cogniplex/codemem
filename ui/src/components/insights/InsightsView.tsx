import { useState } from 'react'
import { ActivityTab } from './ActivityTab'
import { CodeHealthTab } from './CodeHealthTab'
import { SecurityTab } from './SecurityTab'
import { PerformanceTab } from './PerformanceTab'

const tabs = [
  { key: 'activity', label: 'Activity' },
  { key: 'code-health', label: 'Code Health' },
  { key: 'security', label: 'Security' },
  { key: 'performance', label: 'Performance' },
] as const

type TabKey = (typeof tabs)[number]['key']

const tabComponents: Record<TabKey, React.FC> = {
  activity: ActivityTab,
  'code-health': CodeHealthTab,
  security: SecurityTab,
  performance: PerformanceTab,
}

export function InsightsView() {
  const [activeTab, setActiveTab] = useState<TabKey>('activity')
  const TabContent = tabComponents[activeTab]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-zinc-100">Insights</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Enriched analysis from git history, security scans, and performance profiling.
        </p>
      </div>

      <div className="flex gap-1 rounded-lg border border-zinc-800 bg-zinc-900/50 p-1">
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === key
                ? 'bg-zinc-800 text-zinc-100'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <TabContent />
    </div>
  )
}
