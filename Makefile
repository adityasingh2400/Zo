.PHONY: bootstrap bootstrap-render start stop

bootstrap:
	@bash scripts/demo_bootstrap.sh

bootstrap-render:
	@bash scripts/demo_bootstrap.sh --render

start: stop
	@echo "================================="
	@echo "  EMPIRE — Starting services"
	@echo "================================="
	@cd backend && source venv/bin/activate && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --ws-max-size 67108864 --reload &
	@cd dashboard && npm run dev &
	@sleep 2
	@echo ""
	@echo "================================="
	@echo "  Dashboard: http://localhost:5173"
	@echo "  Backend:   http://localhost:8000"
	@echo "  API docs:  http://localhost:8000/docs"
	@echo "================================="
	@echo ""
	@echo "Logs streaming below. Ctrl+C to stop all."
	@wait

stop:
	@lsof -ti :8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti :5173 | xargs kill -9 2>/dev/null || true
	@echo "Stopped all services."
