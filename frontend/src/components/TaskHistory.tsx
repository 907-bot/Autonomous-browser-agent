'use client'

import { motion } from 'framer-motion'
import { History, Trash2, CheckCircle, XCircle, Clock, ExternalLink } from 'lucide-react'
import { useAgentStore } from '@/lib/store'
import { formatTimestamp, cn } from '@/lib/utils'

export default function TaskHistory() {
  const { taskHistory, clearHistory } = useAgentStore()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="w-5 h-5 text-cyan-400" />
          <h3 className="text-lg font-semibold text-white">Task History</h3>
          <span className="px-2 py-1 text-xs font-medium bg-cyan-500/20 text-cyan-400 rounded-full">
            {taskHistory.length}
          </span>
        </div>
        
        {taskHistory.length > 0 && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={clearHistory}
            className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-400 hover:text-rose-400 transition-colors rounded-lg hover:bg-rose-500/10"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </motion.button>
        )}
      </div>

      {/* Empty State */}
      {taskHistory.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-12 text-center"
        >
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-dark-400/50 flex items-center justify-center">
            <History className="w-8 h-8 text-gray-600" />
          </div>
          <p className="text-gray-400 mb-2">No tasks executed yet</p>
          <p className="text-sm text-gray-500">
            Execute a task to see your history here
          </p>
        </motion.div>
      ) : (
        /* Task List */
        <div className="space-y-3">
          {[...taskHistory].reverse().map((task, index) => (
            <motion.div
              key={task.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-4 hover:border-cyan-500/20 transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                {/* Status Icon */}
                <div className={cn(
                  'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center',
                  task.success 
                    ? 'bg-emerald-500/20' 
                    : 'bg-amber-500/20'
                )}>
                  {task.success ? (
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-amber-400" />
                  )}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-medium text-white truncate">
                    {task.task}
                  </h4>
                  <div className="mt-1 flex items-center gap-3 text-xs text-gray-500">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatTimestamp(task.timestamp)}
                    </span>
                    <span className="flex items-center gap-1">
                      <ExternalLink className="w-3 h-3" />
                      {new URL(task.url).hostname}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <span className={cn(
                      'px-2 py-0.5 text-xs font-medium rounded',
                      task.success 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-amber-500/20 text-amber-400'
                    )}>
                      {task.success ? 'Completed' : 'Partial'}
                    </span>
                    <span className="text-xs text-gray-500">
                      {task.steps_completed} steps
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}