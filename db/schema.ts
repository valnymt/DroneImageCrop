import { sql } from "drizzle-orm";
import { integer, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const analyses = sqliteTable("analyses", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
  cropType: text("crop_type").notNull(),
  fieldSizeHectares: real("field_size_hectares").notNull(),
  averageYieldPerPlantKg: real("average_yield_per_plant_kg").notNull(),
  plantCount: integer("plant_count").notNull(),
  cropDensity: real("crop_density").notNull(),
  cropCoverage: real("crop_coverage").notNull(),
  vegetationScore: real("vegetation_score").notNull(),
  healthScore: real("health_score").notNull(),
  estimatedYield: real("estimated_yield").notNull(),
  confidenceScore: real("confidence_score").notNull(),
  imagePath: text("image_path"),
});
