import { useState } from 'react'
import { useLocation, useParams, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ArrowLeftIcon,
  ArrowDownTrayIcon,
  PrinterIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  EyeIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline'

export default function Results() {
  const { id } = useParams()
  const location = useLocation()
  const result = location.state?.result
  const [activeSliceIdx, setActiveSliceIdx] = useState(0)
  const [filenameSearch, setFilenameSearch] = useState('')

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">No Report Data</h2>
          <Link to="/analysis" className="btn-primary">Return to Console</Link>
        </div>
      </div>
    )
  }

  const { predictions, overall_risk, confidence, processing_time } = result
  const sliceImages = result.slice_images || []
  const hasGallery = sliceImages.length > 0

  const activeImage = hasGallery ? sliceImages[activeSliceIdx] : null
  const activeBase64 = activeImage?.image_base64 || result.image_base64

  return (
    <div className="py-4 px-4 min-h-screen bg-slate-950">
      
      {/* Utility Bar */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
        <Link to="/analysis" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
          <ArrowLeftIcon className="w-4 h-4" />
          <span className="text-sm font-medium">Back to Console</span>
        </Link>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${overall_risk === 'High' ? 'bg-red-500 animate-pulse' : overall_risk === 'Moderate' ? 'bg-amber-500' : 'bg-green-500'}`}></div>
            <span className="font-mono text-sm text-white font-bold">RISK: {overall_risk.toUpperCase()}</span>
          </div>
          <button className="btn-secondary flex items-center gap-2 text-xs">
            <PrinterIcon className="w-4 h-4" /> Print
          </button>
          <button className="btn-primary flex items-center gap-2 text-xs">
            <ArrowDownTrayIcon className="w-4 h-4" /> Export
          </button>
        </div>
      </div>

      {/* TOP SECTION: Split Grid */}
      <div className="grid grid-cols-12 gap-4 mb-4">
        
        {/* LEFT: Image Viewer + Thumbnails + Stats (Span 7) */}
        <div className="col-span-12 lg:col-span-7 flex flex-col gap-3">
          {/* Image Viewer */}
          <div className="bg-black rounded border border-slate-800 relative flex items-center justify-center overflow-hidden" style={{ height: '480px' }}>
            {/* DICOM Info */}
            <div className="absolute top-3 left-3 z-10 flex flex-col gap-0.5">
              {activeImage && (
                <>
                  <span className="text-[10px] font-mono text-emerald-500">
                    {activeImage.location}
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400 break-all">
                    {activeImage.filename || 'N/A'}
                  </span>
                </>
              )}
            </div>
            
            {/* Nav Arrows */}
            {hasGallery && sliceImages.length > 1 && (
              <>
                <button 
                  onClick={() => setActiveSliceIdx(Math.max(0, activeSliceIdx - 1))}
                  className="absolute left-1 top-1/2 -translate-y-1/2 z-20 bg-black/60 hover:bg-black/80 p-1.5 rounded-full border border-slate-700 disabled:opacity-30"
                  disabled={activeSliceIdx === 0}
                >
                  <ChevronLeftIcon className="w-4 h-4 text-white" />
                </button>
                <button 
                  onClick={() => setActiveSliceIdx(Math.min(sliceImages.length - 1, activeSliceIdx + 1))}
                  className="absolute right-1 top-1/2 -translate-y-1/2 z-20 bg-black/60 hover:bg-black/80 p-1.5 rounded-full border border-slate-700 disabled:opacity-30"
                  disabled={activeSliceIdx === sliceImages.length - 1}
                >
                  <ChevronRightIcon className="w-4 h-4 text-white" />
                </button>
              </>
            )}

            {/* Main Image */}
            <AnimatePresence mode="wait">
              {activeBase64 ? (
                <motion.img
                  key={activeSliceIdx}
                  src={`data:image/png;base64,${activeBase64}`}
                  alt="Analysis"
                  className="max-h-full max-w-full object-contain"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.15 }}
                />
              ) : (
                <div className="text-slate-600 font-mono text-sm">NO VISUAL DATA</div>
              )}
            </AnimatePresence>

            {/* Slice Counter */}
            {hasGallery && (
              <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded border border-white/10">
                <span className="font-mono text-[10px] text-slate-400">
                  {activeSliceIdx + 1}/{sliceImages.length}
                </span>
              </div>
            )}
          </div>

          {/* Thumbnail Strip */}
          {hasGallery && (
            <div className="bg-slate-900/80 rounded border border-slate-800 p-2">
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {sliceImages.map((slice, idx) => {
                  const locCount = sliceImages.filter(s => s.location === slice.location).length;
                  return (
                  <button
                    key={idx}
                    onClick={() => setActiveSliceIdx(idx)}
                    className={`flex-shrink-0 relative rounded overflow-hidden border-2 transition-all duration-150 ${
                      idx === activeSliceIdx 
                        ? 'border-emerald-500 ring-1 ring-emerald-500/30' 
                        : 'border-slate-700 hover:border-slate-500 opacity-60 hover:opacity-100'
                    }`}
                    style={{ width: '64px', height: '64px' }}
                  >
                    <img 
                      src={`data:image/png;base64,${slice.image_base64}`}
                      alt={slice.location}
                      className="w-full h-full object-cover"
                    />
                    {locCount > 1 && (
                      <div className="absolute top-0 right-0 bg-cyan-600 text-white text-[7px] font-bold px-0.5 rounded-bl">
                        {locCount}
                      </div>
                    )}
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent px-0.5 py-0.5">
                      <span className={`text-[8px] font-mono font-bold ${slice.probability > 0.7 ? 'text-red-400' : 'text-amber-400'}`}>
                        {(slice.probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  </button>
                )})}
              </div>
            </div>
          )}

          {/* Stats Cards - MOVED HERE */}
          <div className="grid grid-cols-4 gap-3">
            <div className="card p-3 bg-slate-900 border-slate-800">
              <div className="label-text text-[10px]">Inference Time</div>
              <div className="value-text text-lg">{processing_time.toFixed(2)}s</div>
            </div>
            <div className="card p-3 bg-slate-900 border-slate-800">
              <div className="label-text text-[10px]">Max Confidence</div>
              <div className="value-text text-lg">{(confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="card p-3 bg-slate-900 border-slate-800">
              <div className="label-text text-[10px]">Findings</div>
              <div className="value-text text-lg">{sliceImages.length} slices</div>
            </div>
            <div className="card p-3 bg-slate-900 border-slate-800">
              <div className="label-text text-[10px]">Modality</div>
              <div className="value-text text-lg">{result.modality || 'CTA'}</div>
            </div>
          </div>
        </div>

        {/* RIGHT: Segmentation Report (Span 5) - Full Height */}
        <div className="col-span-12 lg:col-span-5 flex flex-col">
          <div className="bg-slate-900 border border-slate-800 rounded flex-1 flex flex-col h-full">
            <div className="p-3 border-b border-slate-800">
              <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                Segmentation Report
              </h2>
            </div>
            <div className="overflow-auto flex-1">
              <table className="w-full">
                <thead className="sticky top-0 bg-slate-900/95 backdrop-blur z-10">
                  <tr className="border-b border-slate-800">
                    <th className="text-left py-2 px-3 font-mono text-[10px] text-slate-500">Region</th>
                    <th className="text-right py-2 px-3 font-mono text-[10px] text-slate-500">Prob</th>
                    <th className="text-center py-2 px-3 font-mono text-[10px] text-slate-500">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions
                    .filter(p => p.location !== 'Aneurysm Present')
                    .sort((a, b) => b.probability - a.probability)
                    .map((pred, idx) => {
                      const galleryIdx = sliceImages.findIndex(s => s.location === pred.location)
                      return (
                      <tr 
                        key={idx} 
                        className={`border-b border-slate-800/50 transition-colors cursor-pointer ${
                          pred.detected ? 'bg-red-900/10 hover:bg-red-900/20' : 'hover:bg-white/5'
                        } ${galleryIdx === activeSliceIdx && galleryIdx !== -1 ? 'ring-1 ring-inset ring-emerald-500/30' : ''}`}
                        onClick={() => galleryIdx !== -1 && setActiveSliceIdx(galleryIdx)}
                      >
                        <td className="py-1.5 px-3 text-xs text-slate-300">
                          <div className="flex items-center gap-1">
                            {galleryIdx !== -1 && <EyeIcon className="w-3 h-3 text-emerald-500 flex-shrink-0" />}
                            <span className="truncate">{pred.location}</span>
                          </div>
                        </td>
                        <td className="py-1.5 px-3 text-right font-mono text-xs text-slate-400">
                          {(pred.probability).toFixed(4)}
                        </td>
                        <td className="py-1.5 px-3 text-center">
                          {pred.detected ? (
                            <span className="badge badge-danger text-[10px]">DETECTED</span>
                          ) : (
                            <span className="badge badge-neutral text-[10px]">NORMAL</span>
                          )}
                        </td>
                      </tr>
                    )})}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* BOTTOM: Full-Width Detailed Findings Table */}
      <div className="bg-slate-900 border border-slate-800 rounded">
        <div className="p-3 border-b border-slate-800 flex items-center justify-between gap-4">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex-shrink-0">
            Detailed Findings (Multi-Slice)
          </h2>
          <div className="relative max-w-sm w-full">
            <MagnifyingGlassIcon className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by filename..."
              value={filenameSearch}
              onChange={(e) => setFilenameSearch(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded text-xs font-mono text-white pl-8 pr-3 py-1.5 placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 transition-colors"
            />
            {filenameSearch && (
              <button
                onClick={() => setFilenameSearch('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white text-xs"
              >
                ✕
              </button>
            )}
          </div>
        </div>
        <div className="overflow-auto" style={{ maxHeight: '450px' }}>
          <table className="w-full">
            <thead className="sticky top-0 bg-slate-900/95 backdrop-blur z-10">
              <tr className="border-b border-slate-800">
                <th className="text-left py-2.5 px-4 font-mono text-[10px] text-slate-500 uppercase">Region</th>
                <th className="text-left py-2.5 px-4 font-mono text-[10px] text-slate-500">File Name</th>
                <th className="text-right py-2.5 px-4 font-mono text-[10px] text-slate-500">XY</th>
                <th className="text-right py-2.5 px-4 font-mono text-[10px] text-slate-500">Probability</th>
              </tr>
            </thead>
            <tbody>
              {predictions
                .filter(p => p.detected && p.detailed_coordinates && p.detailed_coordinates.length > 0)
                .flatMap(pred => {
                  const coords = filenameSearch
                    ? pred.detailed_coordinates.filter(d => (d.filename || '').toLowerCase().includes(filenameSearch.toLowerCase()))
                    : pred.detailed_coordinates;
                  return coords.map((detail, i) => {
                    const galleryIdx = sliceImages.findIndex(
                      s => s.location === pred.location && s.slice_z === detail.z
                    );
                    return (
                    <tr 
                      key={`${pred.location}-${i}`} 
                      className={`border-b border-slate-800/40 hover:bg-white/5 cursor-pointer ${
                        galleryIdx === activeSliceIdx && galleryIdx !== -1 ? 'bg-emerald-900/20' : ''
                      }`}
                      onClick={() => galleryIdx !== -1 && setActiveSliceIdx(galleryIdx)}
                    >
                      <td className="py-1.5 px-4 text-xs text-slate-300">
                        {i === 0 ? (
                          <div className="flex items-center gap-1.5">
                            <span className="font-medium">{pred.location}</span>
                            {pred.detailed_coordinates.length > 1 && (
                              <span className="bg-cyan-800 text-cyan-200 text-[9px] px-1 rounded flex-shrink-0">
                                {pred.detailed_coordinates.length} slices
                              </span>
                            )}
                          </div>
                        ) : ''}
                      </td>
                      <td className="py-1.5 px-4 text-cyan-400 font-mono text-xs">
                        {detail.filename || '-'}
                      </td>
                      <td className="py-1.5 px-4 text-right text-slate-400 font-mono text-xs">
                        {detail.x}, {detail.y}
                      </td>
                      <td className="py-1.5 px-4 text-right text-slate-400 font-mono text-xs">
                        {detail.prob.toFixed(4)}
                      </td>
                    </tr>
                  )})
                })
              }
              {predictions.filter(p => p.detected && p.detailed_coordinates && p.detailed_coordinates.length > 0).length === 0 && (
                <tr><td colSpan="4" className="text-center py-6 text-slate-500 text-xs">No multi-slice detections found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
