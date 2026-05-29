'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Loader2, CheckCircle, AlertCircle, Clock, Globe, Settings } from 'lucide-react'
import { useAgentStore } from '@/lib/store'
import { executeTask } from '@/lib/api'
import { cn } from '@/lib/utils'

export default function TaskExecutor() {
  const {
    isExecuting,
    setExecuting,
    currentTask,
    setCurrentTask,
    currentUrl,
    setCurrentUrl,
    status,
    setStatus,
    results,
    setResults,
    headless,
    setHeadless,
    maxSteps,
    setMaxSteps,
    addToHistory,
  } = useAgentStore()

  const [errors, setErrors] = useState<{ task?: string; url?: string }>({})

  const validate = () => {
    const newErrors: { task?: string; url?: string } = {}
    
    if (!currentTask.trim()) {
      newErrors.task = 'Task description is required'
    }
    
    if (!currentUrl.trim()) {
      newErrors.url = 'Starting URL is required'
    } else if (!isValidUrl(currentUrl)) {
      newErrors.url = 'Please enter a valid URL'
    }
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const isValidUrl = (url: string) => {
    try {
      new URL(url)
      return true
    } catch {
      return false
    }
  }

  const handleExecute = async () => {
    if (!validate()) return

    setExecuting(true)
    setStatus('🚀 Starting task execution...')
    setResults('')

    try {
      const response = await executeTask({
        task: currentTask,
        url: currentUrl,
        headless,
        max_steps: maxSteps,
      })

      setStatus(response.status_text)
      setResults(response.results_json)

      // Add to history
      addToHistory({
        id: Date.now().toString(),
        task: currentTask,
        url: currentUrl,
        status: 'completed',
        success: response.success ?? true,
        steps_completed: 0,
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      setStatus(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`)
      setResults(JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }, null, 2))
    } finally {
      setExecuting(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Task Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <label className="block text-sm font-medium text-gray-300 mb-2">
          📝 Task Description
        </label>
        <textarea
          value={currentTask}
          onChange={(e) => {
            setCurrentTask(e.target.value)
            setErrors((prev) => ({ ...prev, task: undefined }))
          }}
          placeholder="Example: Search for flights from NYC to London on December 20"
          rows={3}
          className={cn(
            'w-full px-4 py-3 bg-dark-400/50 rounded-lg border text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 transition-all resize-none',
            errors.task ? 'border-rose-500' : 'border-white/10'
          )}
        />
        {errors.task && (
          <p className="mt-1 text-sm text-rose-400">{errors.task}</p>
        )}
      </motion.div>

      {/* URL Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <label className="block text-sm font-medium text-gray-300 mb-2">
          🌐 Starting URL
        </label>
        <div className="relative">
          <Globe className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
          <input
            type="text"
            value={currentUrl}
            onChange={(e) => {
              setCurrentUrl(e.target.value)
              setErrors((prev) => ({ ...prev, url: undefined }))
            }}
            placeholder="https://www.google.com"
            className={cn(
              'w-full pl-11 pr-4 py-3 bg-dark-400/50 rounded-lg border text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 transition-all',
              errors.url ? 'border-rose-500' : 'border-white/10'
            )}
          />
        </div>
        {errors.url && (
          <p className="mt-1 text-sm text-rose-400">{errors.url}</p>
        )}
      </motion.div>

      {/* Settings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-gray-300">Settings</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {/* Headless Toggle */}
          <div className="flex items-center justify-between p-4 bg-dark-400/30 rounded-lg">
            <div>
              <p className="text-sm text-white">🎭 Headless Mode</p>
              <p className="text-xs text-gray-500">Run browser without UI</p>
            </div>
            <button
              onClick={() => setHeadless(!headless)}
              className={cn(
                'relative w-12 h-6 rounded-full transition-colors',
                headless ? 'bg-cyan-500' : 'bg-gray-600'
              )}
            >
              <motion.div
                animate={{ x: headless ? 24 : 0 }}
                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full"
              />
            </button>
          </div>

          {/* Max Steps */}
          <div>
            <label className="block text-sm text-white mb-2">
              ⏱️ Max Steps: {maxSteps}
            </label>
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={maxSteps}
              onChange={(e) => setMaxSteps(parseInt(e.target.value))}
              className="w-full accent-cyan-500"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>5</span>
              <span>100</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Execute Button */}
      <motion.button
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        onClick={handleExecute}
        disabled={isExecuting}
        className={cn(
          'w-full py-4 rounded-xl font-semibold text-white transition-all duration-300 flex items-center justify-center gap-2',
          isExecuting
            ? 'bg-gray-600 cursor-not-allowed'
            : 'bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40'
        )}
      >
        {isExecuting ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Executing...
          </>
        ) : (
          <>
            <Play className="w-5 h-5" />
            Execute Task
          </>
        )}
      </motion.button>

      {/* Status Display */}
      {status && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-gray-300">Status</span>
          </div>
          <div className="prose prose-invert prose-sm max-w-none">
            <pre className="bg-dark-400/50 rounded-lg p-4 text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap font-mono">
              {status}
            </pre>
          </div>
        </motion.div>
      )}

      {/* Results Display */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            {results.includes('"success": true') || results.includes('"success":true') ? (
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            ) : (
              <AlertCircle className="w-4 h-4 text-amber-400" />
            )}
            <span className="text-sm font-medium text-gray-300">Results (JSON)</span>
          </div>
          <pre className="bg-dark-400/50 rounded-lg p-4 text-xs text-gray-400 overflow-x-auto max-h-96">
            {results}
          </pre>
        </motion.div>
      )}
    </div>
  )
}