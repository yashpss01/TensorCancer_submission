import polars as pl, numpy as np, sys, shutil
from pipeline_core import assign_one_to_one, decide
W="/Users/akanksha/aws/TensorCancer_submission/work/"; N=W+"norm/"
K=[1,2,3,4,5,7,9,11,13,21]
RMIN=float(sys.argv[1]); FLOOR=float(sys.argv[2]); OUT=sys.argv[3]
all_s1=pl.read_parquet(N+"test_s1.parquet",columns=["entity_id","country"])
lists=[]; stats=[]
for c in ["France","India","US"]:
    sc=pl.read_parquet(W+f"scored_test_{c}.parquet",columns=["a","b","p2","r_core_tset","aa_jac"])
    s1=pl.scan_parquet(N+"test_s1.parquet").filter(pl.col("country")==c).select(pl.col("addr_first_num").alias("fa"),"entity_id").collect().with_row_index("a").with_columns(pl.col("a").cast(pl.Int32))
    q=pl.concat([pl.scan_parquet(N+f"test_s{i}.parquet").filter(pl.col("country")==c).select(pl.col("addr_first_num").alias("fb"),"entity_id").collect() for i in (2,3)]).with_row_index("b").with_columns(pl.col("b").cast(pl.Int32))
    sc=sc.join(s1.select("a","fa"),on="a").join(q.select("b","fb"),on="b")
    d=pl.col("fb").str.slice(0,9).cast(pl.Int64,strict=False)-pl.col("fa").str.slice(0,9).cast(pl.Int64,strict=False)
    sc=sc.with_columns(d.alias("d")).with_columns((pl.col("d").is_in(K)&(pl.col("r_core_tset")>=RMIN)&(pl.col("aa_jac")>=0.6)).fill_null(False).alias("kp"))
    sc=sc.with_columns(pl.when(pl.col("kp")).then(pl.max_horizontal("p2",pl.lit(FLOOR))).otherwise(pl.col("p2")).alias("pk"))
    base=decide(assign_one_to_one(sc,"p2"),"p2","expected")
    pred=decide(assign_one_to_one(sc,"pk"),"pk","expected")
    n1=s1.shape[0]
    stats.append((c,base.shape[0]/n1,pred.shape[0]/n1,int(sc["kp"].sum())))
    p=pred.with_columns(pl.Series("source1_entity_id",s1["entity_id"].to_numpy()[pred["a"].to_numpy()]),pl.Series("q_id",q["entity_id"].to_numpy()[pred["b"].to_numpy()]))
    lists.append(p.group_by("source1_entity_id").agg(pl.col("q_id").sort().str.join(",").alias("matched_entity_ids")))
out=all_s1.select(pl.col("entity_id").alias("source1_entity_id")).join(pl.concat(lists),on="source1_entity_id",how="left",maintain_order="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
with open(OUT+"/matching_results.tsv","w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s,i in out.iter_rows(): f.write(f"{s}\t{i}\n")
shutil.copyfile(W+"../output_v5/candidate_pairs.tsv",OUT+"/candidate_pairs.tsv")
for s in stats: print(f"{s[0]}: accepted/S1 v5 {s[1]:.3f} -> variant {s[2]:.3f}; +k pairs boosted {s[3]}")
