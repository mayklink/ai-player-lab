"""Executa inferência real por API remota com opt-in; pode consumir créditos.

Mede contrato/latência numa fixture sintética; não valida compreensão de Mario.
"""
from pathlib import Path
import argparse
import json
import statistics
import time
from ai_player_lab.memory import EpisodeBuffer
from ai_player_lab.perception import VisualPipeline
from ai_player_lab.providers import MultimodalReviewer
from ai_player_lab.diagnostics import fixture_frame, machine_report
from ai_player_lab.types import Decision


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',default=None)
    parser.add_argument('--model',default=None)
    parser.add_argument('--allow-api-calls',action='store_true',help='Autoriza envio e possíveis cobranças')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--out',type=Path,default=Path('runs/vlm-benchmark.json'))
    args=parser.parse_args()
    if not args.allow_api_calls: parser.error('Use --allow-api-calls para autorizar inferência remota')
    if not 1<=args.runs<=10: parser.error('--runs deve ser 1..10')
    first,box=fixture_frame(0); pipeline=VisualPipeline(first,box); buffer=EpisodeBuffer()
    for i in range(12):
        frame,_=fixture_frame(i); obs=pipeline.process(frame,1+i*.25)
        buffer.add(frame,pipeline.last_static,obs,Decision('right',obs.captured_at,obs.seq))
    job=buffer.snapshot('synthetic_observation_no_death',4)
    client=MultimodalReviewer(args.base_url,args.model,timeout_s=90,max_calls=args.runs)
    results=[]
    try:
        for _ in range(args.runs):
            start=time.monotonic()
            try:
                response=client(job)
                results.append({'seconds':time.monotonic()-start,'schema_valid':True,'review':response['review']})
            except Exception as exc:
                results.append({'seconds':time.monotonic()-start,'schema_valid':False,'error':type(exc).__name__})
    finally: client.close()
    report={'model':client.model,'api_calls':client.calls,'fixture':'synthetic_not_Mario','runs':results,
            'median_seconds':statistics.median(x['seconds'] for x in results),
            'valid_fraction':sum(x['schema_valid'] for x in results)/len(results),
            'semantic_accuracy':'NOT_MEASURED','machine':machine_report()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if all(x['schema_valid'] for x in results) else 1


if __name__=='__main__': raise SystemExit(main())
