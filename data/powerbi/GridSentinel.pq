// GridSentinel — Power Query M script for Power BI Desktop
// Paste this into a new query via Home → Transform data → New Source → Blank Query
// → Home → Advanced Editor.
//
// Then set DataFolder to the absolute path of your data/powerbi/ directory.
// Each let-binding below becomes a separate query/table in the model.

let
  DataFolder = "C:/path/to/GridSentinel/data/powerbi",  // ← edit this
  fact_events = let
    Source = Csv.Document(
      File.Contents(DataFolder & "/fact_events.csv"),
      [Delimiter=",", Columns=12, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"tick", Int64.Type}, {"kind", type text}, {"scope", type text}, {"target", type text}, {"severity", type text}, {"benign", type logical}, {"parent_event_id", type text}, {"node_id", type text}, {"rack_id", type text}, {"zone_id", type text}, {"feed_id", type text}, {"spine_id", type text}})
  in Typed,
  fact_incidents = let
    Source = Csv.Document(
      File.Contents(DataFolder & "/fact_incidents.csv"),
      [Delimiter=",", Columns=12, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"incident_id", type text}, {"opened_tick", Int64.Type}, {"resolved_tick", Int64.Type}, {"member_count", Int64.Type}, {"root_kind", type text}, {"scope", type text}, {"severity_max", type text}, {"all_benign", type logical}, {"any_benign", type logical}, {"duration_ticks", Int64.Type}, {"cluster_id", Int64.Type}, {"cluster_label", type text}})
  in Typed,
  fact_hourly_kpi = let
    Source = Csv.Document(
      File.Contents(DataFolder & "/fact_hourly_kpi.csv"),
      [Delimiter=",", Columns=5, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"tick", Int64.Type}, {"fleet_kw", type number}, {"active_gpu_count", Int64.Type}, {"gpu_hours", type number}, {"idle_gpu_count", Int64.Type}})
  in Typed,
  fact_clusters = let
    Source = Csv.Document(
      File.Contents(DataFolder & "/fact_clusters.csv"),
      [Delimiter=",", Columns=6, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"cluster_id", Int64.Type}, {"cluster_label", type text}, {"count", Int64.Type}, {"critical_count", Int64.Type}, {"all_benign_ratio", type number}, {"mean_members", type number}})
  in Typed,
  dim_fleet = let
    Source = Csv.Document(
      File.Contents(DataFolder & "/dim_fleet.csv"),
      [Delimiter=",", Columns=5, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"node_id", type text}, {"rack_id", type text}, {"zone_id", type text}, {"feed_id", type text}, {"spine_id", type text}})
  in Typed,
  dim_time = let
    Source = Csv.Document(
      File.Contents(DataFolder & "/dim_time.csv"),
      [Delimiter=",", Columns=6, Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"tick", Int64.Type}, {"datetime", type datetime}, {"date", type datetime}, {"hour", Int64.Type}, {"day_of_week", type text}, {"is_weekend", type logical}})
  in Typed
in
  // Returns the last table; reference any other table by its name in the
  // queries pane after the script runs.
  fact_incidents
