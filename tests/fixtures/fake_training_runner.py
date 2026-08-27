"""Small subprocess used to exercise the Qt training lifecycle."""

import json
import signal
import sys
import time
from pathlib import Path


PREFIX = '@@FILESPROCESS_TRAIN@@'


def emit(event_type, **payload):
    print(
        PREFIX + json.dumps({'type': event_type, **payload}),
        flush=True,
    )


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    run_dir = (
        Path(job['output_root']) / job['project_name'] / job['run_name']
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    emit('initializing', expected_run_dir=str(run_dir))
    emit(
        'started', save_dir=str(run_dir),
        epochs=job['parameters'].get('epochs', 2), device='test',
    )
    if job['run_name'].startswith('slow'):
        while True:
            print('fake runner waiting', flush=True)
            time.sleep(0.1)

    epochs = int(job['parameters'].get('epochs', 2))
    for epoch in range(1, epochs + 1):
        emit(
            'epoch', epoch=epoch, epochs=epochs,
            progress=epoch / epochs * 100,
            metrics={
                'train/box_loss': 1.0 / epoch,
                'metrics/mAP50-95(P)': epoch / epochs * 0.8,
            },
        )
    emit('finalizing', save_dir=str(run_dir))
    (run_dir / 'weights').mkdir(exist_ok=True)
    (run_dir / 'weights' / 'best.pt').write_bytes(b'fake-model')
    (run_dir / 'args.yaml').write_text(
        '\n'.join((
            f"task: {job['ultralytics_task']}",
            f"model: {job['model']}",
            f"data: {job['dataset_yaml']}",
            f"project: {Path(job['output_root']) / job['project_name']}",
            f"name: {job['run_name']}",
            f"epochs: {epochs}",
            'imgsz: 640',
            'batch: 2',
            'optimizer: auto',
            'device: test',
        )) + '\n',
        encoding='utf-8',
    )
    (run_dir / 'results.csv').write_text(
        'epoch,time,metrics/mAP50-95(P),train/box_loss\n'
        '1,1.0,0.4,1.0\n'
        f'{epochs},2.0,0.8,0.5\n',
        encoding='utf-8',
    )
    Path(run_dir / 'training_request.json').write_text(
        json.dumps(job, indent=2), encoding='utf-8'
    )
    emit('completed', save_dir=str(run_dir))
    return 0


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    sys.exit(main())
