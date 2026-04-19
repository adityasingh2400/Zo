import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import StageView from './StageView'

// Tiny path-based router — keeps the dashboard router-free (no react-router
// dep) while still letting the operator hit /stage on the demo Mac for the
// fullscreen TikTok Shop overlay. Two routes are all the demo needs:
//   /        → existing operator dashboard (App.jsx)
//   /stage   → demo surface (StageView.jsx) — fullscreen, hotkey-driven
//
// Switches re-evaluated on popstate so back/forward in the browser still
// flips between views without a hard reload.
function Root() {
  const [path, setPath] = React.useState(() => window.location.pathname);
  React.useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);
  const isStage = path === '/stage' || path.startsWith('/stage/');
  return isStage ? <StageView /> : <App />;
}

ReactDOM.createRoot(document.getElementById('root')).render(<Root />)
