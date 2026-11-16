import { useEffect, useMemo, useRef, useState } from 'react';

const formatStatus = (status) => {
  if (status === 'BERGERAK') return 'Bergerak';
  if (status === 'TIDAK AKTIF') return 'Tidak aktif';
  if (status === 'PERLU OBSERVASI') return 'Perlu observasi';
  return 'Mengamati';
};

function PigMark() {
  return <div className="pig-mark" aria-hidden="true">P</div>;
}

function CameraCard({ name, stats, onStart, onStop, busy, onInactive }) {
  const [pigCount, setPigCount] = useState(stats?.max || '');
  const [message, setMessage] = useState('');
  const previousStatuses = useRef({});
  const activities = Object.entries(stats?.activities || {}).sort(
    ([a], [b]) => Number(a) - Number(b),
  );

  useEffect(() => {
    if (stats?.active && stats.max) setPigCount(String(stats.max));
  }, [stats?.active, stats?.max]);

  useEffect(() => {
    const nextStatuses = {};
    for (const [id, activity] of Object.entries(stats?.activities || {})) {
      const current = activity.status || 'MENGAMATI';
      const previous = previousStatuses.current[id];
      if (previous && previous !== 'TIDAK AKTIF' && current === 'TIDAK AKTIF') {
        onInactive(name, id, Number(activity.inactive_seconds || 0));
      }
      nextStatuses[id] = current;
    }
    previousStatuses.current = nextStatuses;
  }, [name, onInactive, stats?.activities]);

  const start = async () => {
    const count = Number(pigCount);
    if (!Number.isInteger(count) || count < 1 || count > 300) {
      setMessage('Masukkan jumlah babi antara 1 sampai 300.');
      return;
    }
    try {
      await onStart(name, count);
      setMessage('Monitoring dimulai.');
    } catch (error) {
      setMessage(error.message || 'Tidak dapat memulai monitoring.');
    }
  };

  const stop = async () => {
    try {
      await onStop(name);
      setMessage('Monitoring dihentikan.');
    } catch (error) {
      setMessage(error.message || 'Tidak dapat menghentikan monitoring.');
    }
  };

  const online = Boolean(stats?.online);

  return (
    <section className="camera-card">
      <div className="camera-heading">
        <div>
          <span className="overline">SUMBER VIDEO</span>
          <h2>{name}</h2>
        </div>
        <span className={`connection ${online ? 'is-online' : 'is-offline'}`}>
          <i />{online ? (stats?.active ? 'Memantau' : 'Siap') : 'Offline'}
        </span>
      </div>

      <div className="video-frame">
        <img src={`/stream/${encodeURIComponent(name)}`} alt={`Live ${name}`} />
        <span className="live-label"><i /> LIVE</span>
        {!online && <div className="video-message">Menunggu sumber video…</div>}
      </div>

      <div className="stat-grid">
        <div className="stat-tile">
          <span className="stat-icon pig-icon">♧</span>
          <div><span className="stat-label">Terdeteksi</span><strong>{stats?.count ?? 0}<small> ekor</small></strong></div>
        </div>
        <div className="stat-tile">
          <span className="stat-icon speed-icon">↗</span>
          <div><span className="stat-label">Kecepatan</span><strong>{stats?.fps ?? 0}<small> FPS</small></strong></div>
        </div>
      </div>

      <div className="monitor-controls">
        <label htmlFor={`pigs-${name}`}>Jumlah babi di kandang</label>
        <div className="control-line">
          <input
            id={`pigs-${name}`}
            type="number"
            min="1"
            max="300"
            placeholder="Contoh: 5"
            value={pigCount}
            onChange={(event) => setPigCount(event.target.value)}
          />
          <button className="button-primary" onClick={start} disabled={busy}>
            {busy ? 'Memproses…' : stats?.active ? 'Terapkan ulang' : 'Mulai monitoring'}
          </button>
          {stats?.active && <button className="button-secondary" onClick={stop} disabled={busy}>Berhenti</button>}
        </div>
        {message && <p className="inline-message" role="status">{message}</p>}
        {stats?.active && stats.max > 0 && (
          <p className="detection-note">
            {stats.count < stats.max
              ? `${stats.count} dari ${stats.max} terdeteksi · sebagian mungkin tertutup`
              : `Semua ${stats.max} babi terdeteksi`}
          </p>
        )}
      </div>

      <div className="activity-section">
        <div className="section-heading">
          <div><span className="overline">PEMANTAUAN</span><h3>Aktivitas individu</h3></div>
          <span className="id-count">{activities.length} ID</span>
        </div>
        {activities.length === 0 ? (
          <div className="empty-state">Mulai monitoring untuk melihat aktivitas setiap babi.</div>
        ) : (
          <div className="activity-list">
            {activities.map(([id, activity]) => {
              const status = activity.status || 'MENGAMATI';
              const movement = Number(activity.movement_px_per_frame ?? activity.movement_ratio ?? 0);
              const total = Number(activity.total_movement_px || 0);
              const inactive = Math.floor(Number(activity.inactive_seconds || 0));
              return (
                <div className="activity-row" key={id}>
                  <div className="pig-id"><span className="id-avatar">{id}</span><div><strong>Babi {id}</strong><small>{movement.toFixed(2)} px/frame · total {total.toFixed(1)} px</small></div></div>
                  <div className="activity-meta"><span className={`activity-status ${status.toLowerCase().replaceAll(' ', '-')}`}>{formatStatus(status)}</span><small>Diam {Math.floor(inactive / 60)}m {inactive % 60}s</small></div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

export default function App() {
  const [cameraStats, setCameraStats] = useState({});
  const [apiState, setApiState] = useState('connecting');
  const [busyCamera, setBusyCamera] = useState('');
  const [toasts, setToasts] = useState([]);
  const cameras = useMemo(() => Object.entries(cameraStats), [cameraStats]);

  const notifyInactive = (camera, pigId, seconds) => {
    const id = `${camera}-${pigId}-${Date.now()}`;
    const duration = Math.floor(seconds);
    const timeText = duration >= 60
      ? `${Math.floor(duration / 60)} menit`
      : `${duration} detik`;
    setToasts((current) => [...current, { id, title: `Babi ${pigId} tidak aktif`, detail: `${camera} · Tidak aktif selama ${timeText}` }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 6500);
  };

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const response = await fetch('/api/stats', { cache: 'no-store' });
        if (!response.ok) throw new Error('API backend tidak merespons.');
        const result = await response.json();
        if (alive) {
          setCameraStats(result);
          setApiState('online');
        }
      } catch {
        if (alive) setApiState('offline');
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 1200);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const runAction = async (name, action, suffix = '') => {
    setBusyCamera(name);
    try {
      const response = await fetch(`/api/${action}/${encodeURIComponent(name)}${suffix}`, { method: 'POST' });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || 'Permintaan gagal.');
      }
    } finally {
      setBusyCamera('');
    }
  };

  const startCamera = (name, count) => runAction(name, 'start', `?n=${count}`);
  const stopCamera = (name) => runAction(name, 'stop');
  const totals = cameras.reduce((sum, [, stats]) => ({
    pigs: sum.pigs + (stats.count || 0),
    active: sum.active + (stats.active ? 1 : 0),
    online: sum.online + (stats.online ? 1 : 0),
  }), { pigs: 0, active: 0, online: 0 });

  return (
    <main>
      <div className="toast-stack" aria-live="polite" aria-atomic="false">
        {toasts.map((toast) => (
          <div className="toast toast-inactive" key={toast.id} role="status">
            <span className="toast-icon">!</span>
            <span><strong>{toast.title}</strong><small>{toast.detail}</small></span>
            <button className="toast-close" aria-label="Tutup notifikasi" onClick={() => setToasts((items) => items.filter((item) => item.id !== toast.id))}>×</button>
          </div>
        ))}
      </div>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand"><PigMark /><div><h1>PigWatch</h1><p>Sistem pemantauan kandang cerdas</p></div></div>
          <div className={`api-indicator ${apiState}`}><span />{apiState === 'online' ? 'API terhubung' : apiState === 'offline' ? 'API terputus' : 'Menghubungkan API'}</div>
        </div>
      </header>

      <div className="page-shell">
        <section className="welcome-row">
          <div><span className="overline">DASHBOARD MONITORING</span><h2>Pantau kondisi kandang</h2><p>Deteksi, tracking, dan aktivitas babi dalam satu tampilan.</p></div>
          <div className="tech-chip"><span>YOLO</span><b>·</b><span>Optical Flow</span><b>·</b><span>Real-time</span></div>
        </section>

        <section className="overview" aria-label="Ringkasan monitoring">
          <div className="overview-card"><span className="overview-symbol">◉</span><div><span>KAMERA ONLINE</span><strong>{totals.online}<small> / {cameras.length}</small></strong></div></div>
          <div className="overview-card"><span className="overview-symbol green">●</span><div><span>MONITORING AKTIF</span><strong>{totals.active}<small> kamera</small></strong></div></div>
          <div className="overview-card"><span className="overview-symbol gold">♧</span><div><span>BABI TERDETEKSI</span><strong>{totals.pigs}<small> ekor</small></strong></div></div>
        </section>

        <section className="camera-section">
          <div className="section-title"><div><h2>Kamera kandang</h2><p>{cameras.length} sumber video terkonfigurasi</p></div><span className="refresh-label"><span className="refresh-dot" /> Diperbarui otomatis</span></div>
          {cameras.length === 0 ? (
            <div className="no-cameras">Belum ada kamera. Atur sumber video melalui variabel <code>CAMERAS</code> saat menjalankan backend.</div>
          ) : (
            <div className="camera-grid">{cameras.map(([name, stats]) => <CameraCard key={name} name={name} stats={stats} onStart={startCamera} onStop={stopCamera} busy={busyCamera === name} onInactive={notifyInactive} />)}</div>
          )}
        </section>

        <footer>PigWatch <span>·</span> Deteksi berbasis YOLO dan pengukuran gerak optical flow</footer>
      </div>
    </main>
  );
}
