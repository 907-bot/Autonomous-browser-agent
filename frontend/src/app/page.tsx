'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Brain, Info, X } from 'lucide-react'
import Hero from '@/components/Hero'
import TaskExecutor from '@/components/TaskExecutor'
import TaskPlanner from '@/components/TaskPlanner'
import TaskHistory from '@/components/TaskHistory'
import AgentStatus from '@/components/AgentStatus'
import About from '@/components/About'
import DemoMode from '@/components/DemoMode'
import { cn } from '@/lib/utils'
import { getHealth } from '@/lib/api'

type Tab = 'execute' | 'planner' | 'history' | 'status' | 'about'

const tabs = [
  { id: 'execute' as Tab, label: '🚀 Execute', icon: Play },
  { id: 'planner' as Tab, label: '🔍 Planner', icon: Brain },
  { id: 'history' as Tab, label: '📜 History', icon: null },
  { id: 'status' as Tab, label: '📊 Status', icon: null },
  { id: 'about' as Tab, label: 'ℹ️ About', icon: Info },
]

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('execute')
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null)

  useEffect(() => {
    const checkApi = async () => {
      try {
        const health = await getHealth()
        setApiAvailable(true)
      } catch {
        setApiAvailable(false)
      }
    }
    checkApi()
  }, [])

  const renderContent = () => {
    switch (activeTab) {
      case 'execute':
        return <TaskExecutor />
      case 'planner':
        return <TaskPlanner />
      case 'history':
        return <TaskHistory />
      case 'status':
        return <AgentStatus />
      case 'about':
        return <About />
      default:
        return <TaskExecutor />
    }
  }

  // Show loading state
  if (apiAvailable === null) {
    return (
      <div className="min-h-screen bg-dark-500 flex items-center justify-center">
        <motion.div
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="text-gray-400"
        >
          Loading...
        </motion.div>
      </div>
    )
  }

  // Show demo mode if API is not available
  if (!apiAvailable) {
    return <DemoMode />
  }

  return (
    <div className="min-h-screen bg-dark-500">
      {/* Hero Section */}
      <Hero />

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 pb-16 sm:px-6 lg:px-8">
        {/* Tab Navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex flex-wrap gap-2 p-1.5 bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all',
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white shadow-lg'
                    : 'text-gray-400 hover:text-white hover:bg-dark-400/50'
                )}
              >
                {tab.icon && <tab.icon className="w-4 h-4" />}
                {tab.label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-sm text-gray-500">
              Built with ❤️ using MAYINI, Playwright, and Vision Transformers
            </p>
            <p className="text-sm text-gray-500">
              © 2024 | Autonomous Browser Agent Project
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}