import os
import time
import pandas as pd
import numpy as np
from alphagenome.models import dna_client, dna_output
from alphagenome.data import genome

# Load model
api_key = os.environ.get('ALPHAGENOME_API_KEY')
dna_model = dna_client.create(api_key=api_key)
print("Model loaded")

# File paths
COORDS_FILE     = os.path.expanduser('~/Downloads/GeneBayes-main/gene_coordinates.tsv')
CHECKPOINT_FILE = os.path.expanduser('~/Downloads/GeneBayes-main/test5_checkpoint.tsv')
OUTPUT_FILE     = os.path.expanduser('~/Downloads/GeneBayes-main/test5_features.tsv')

# Load gene coordinates — 5 genes only
gene_df = pd.read_csv(COORDS_FILE, sep='\t')
gene_df = gene_df.dropna(subset=['start', 'end']).head(5)
print(f"Testing on {len(gene_df)} genes")

# Load checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    checkpoint = pd.read_csv(CHECKPOINT_FILE, sep='\t')
    completed = set(checkpoint['ensg'].tolist())
    all_results = checkpoint.to_dict('records')
    print(f"Resuming from checkpoint: {len(completed)} genes done")
else:
    completed = set()
    all_results = []
    print("Starting fresh")

remaining = gene_df[~gene_df['ensg'].isin(completed)]
print(f"Genes remaining: {len(remaining)}")

CONFIG = {
    'RNA_SEQ':           {'type': dna_output.OutputType.RNA_SEQ,           'attr': 'rna_seq',           'res': 1,    'thresh': 1.0,    'stranded': True},
    'ATAC':              {'type': dna_output.OutputType.ATAC,               'attr': 'atac',              'res': 1,    'thresh': 2.0,    'stranded': False},
    'CAGE':              {'type': dna_output.OutputType.CAGE,               'attr': 'cage',              'res': 1,    'thresh': 5.0,    'stranded': True},
    'DNASE':             {'type': dna_output.OutputType.DNASE,              'attr': 'dnase',             'res': 1,    'thresh': 2.0,    'stranded': False},
    'CHIP_TF':           {'type': dna_output.OutputType.CHIP_TF,            'attr': 'chip_tf',           'res': 128,  'thresh': 1000.0, 'stranded': False},
    'CHIP_HISTONE':      {'type': dna_output.OutputType.CHIP_HISTONE,       'attr': 'chip_histone',      'res': 128,  'thresh': 2.0,    'stranded': False},
    'SPLICE_SITES':      {'type': dna_output.OutputType.SPLICE_SITES,       'attr': 'splice_sites',      'res': 1,    'thresh': 0.1,    'stranded': True},
    'SPLICE_SITE_USAGE': {'type': dna_output.OutputType.SPLICE_SITE_USAGE,  'attr': 'splice_site_usage', 'res': 1,    'thresh': 0.1,    'stranded': True},
    'SPLICE_JUNCTIONS':  {'type': dna_output.OutputType.SPLICE_JUNCTIONS,   'attr': 'splice_junctions',  'res': 64,   'thresh': 1.0,    'stranded': True},
    'CONTACT_MAPS':      {'type': dna_output.OutputType.CONTACT_MAPS,       'attr': 'contact_maps',      'res': 2048, 'thresh': 0.5,    'stranded': False},
    'PROCAP':            {'type': dna_output.OutputType.PROCAP,             'attr': 'procap',            'res': 1,    'thresh': 5.0,    'stranded': True},
}

for i, (_, row) in enumerate(remaining.iterrows()):
    print(f"\nProcessing {row['ensg']}...")
    try:
        mb_int = genome.Interval(
            row['chrom'], int(row['start']), int(row['end']), strand=row['strand']
        ).resize(dna_client.SEQUENCE_LENGTH_1MB)

        gene_feat = {'ensg': row['ensg']}

        for key, cfg in CONFIG.items():
            try:
                output = dna_model.predict_interval(
                    mb_int,
                    requested_outputs=[cfg['type']],
                    ontology_terms=None
                )
                tdata = getattr(output, cfg['attr'])

                if cfg['stranded']:
                    if row['strand'] == '+':
                        tdata = tdata.filter_to_positive_strand()
                    elif row['strand'] == '-':
                        tdata = tdata.filter_to_negative_strand()

                vals = tdata.values
                res = cfg['res']
                rel_start = max(0, (int(row['start']) - mb_int.start) // res)
                rel_end = min(vals.shape[0], (int(row['end']) - mb_int.start) // res)
                if res > 1:
                    rel_end = max(rel_start + 1, rel_end)

                if rel_end > rel_start:
                    if key == 'CONTACT_MAPS':
                        region_vals = vals[rel_start:rel_end, rel_start:rel_end, :]
                    else:
                        region_vals = vals[rel_start:rel_end, :]

                    if region_vals.size > 0:
                        gene_feat[f'{key}_mean'] = float(np.mean(region_vals))
                        gene_feat[f'{key}_max']  = float(np.max(region_vals))
                        gene_feat[f'{key}_std']  = float(np.std(region_vals))

                        if key == 'CHIP_TF':
                            gene_feat[f'{key}_breadth_1000'] = int(np.sum(region_vals > 1000))
                            gene_feat[f'{key}_breadth_2000'] = int(np.sum(region_vals > 2000))
                        else:
                            gene_feat[f'{key}_breadth'] = int(np.sum(region_vals > cfg['thresh']))

                del output
                print(f"  {key}: SUCCESS, mean={gene_feat.get(f'{key}_mean', 'N/A')}")
                time.sleep(0.5)

            except Exception as e:
                print(f"  {key} FAILED: {e}")
                continue

        all_results.append(gene_feat)

    except Exception as e:
        print(f"Gene {row['ensg']} FAILED: {e}")
        continue

# Save output
final_df = pd.DataFrame(all_results)
final_df.to_csv(OUTPUT_FILE, sep='\t', index=False)
print(f"\nDone. Features extracted for {len(final_df)} genes.")
print(f"Saved to {OUTPUT_FILE}")
print(final_df.to_string())
