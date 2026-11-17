type Reply = Record<string, any>;
interface Task { id: number; job: Record<string, unknown>; resolve(value: Reply): void; reject(error: Error): void }
interface Slot { worker: Worker; ready: boolean; task?: Task }

/** Small CPU pool. Tasks are retained in FIFO order and never superseded. */
export class PoseWorkerPool {
  private slots: Slot[] = [];
  private queue: Task[] = [];
  private nextId = 0;
  private closed = false;
  constructor(baseUrl: string, count = Math.min(3, Math.max(1, (navigator.hardwareConcurrency || 4) - 1))) {
    for (let i = 0; i < count; i++) {
      const worker = new Worker(new URL("./pose.worker.ts", import.meta.url), { type: "module" });
      const slot: Slot = { worker, ready: false };
      this.slots.push(slot);
      worker.onmessage = ({ data }) => {
        if (this.closed) return;
        if (data.type === "ready") { slot.ready = true; this.pump(); }
        if (data.type === "result" && slot.task && slot.task.id === data.id) {
          const task = slot.task; slot.task = undefined; task.resolve(data.result); this.pump();
        }
        if (data.type === "error") this.dispose(new Error(data.message));
      };
      worker.onerror = () => this.dispose(new Error("The CPU pose worker stopped. Toggle fitted pose to retry."));
      worker.postMessage({ type: "init", baseUrl });
    }
  }
  execute(job: Record<string, unknown>): Promise<Reply> {
    if (this.closed) return Promise.reject(new Error("Pose worker pool is closed"));
    return new Promise((resolve, reject) => { this.queue.push({ id: this.nextId++, job, resolve, reject }); this.pump(); });
  }
  private pump() {
    for (const slot of this.slots) {
      if (!slot.ready || slot.task || !this.queue.length) continue;
      slot.task = this.queue.shift()!;
      slot.worker.postMessage({ type: "job", id: slot.task.id, job: slot.task.job });
    }
  }
  dispose(error = new Error("Pose fitting cancelled")) {
    if (this.closed) return; this.closed = true;
    for (const task of this.queue.splice(0)) task.reject(error);
    for (const slot of this.slots) { slot.task?.reject(error); slot.worker.terminate(); }
    this.slots = [];
  }
}
