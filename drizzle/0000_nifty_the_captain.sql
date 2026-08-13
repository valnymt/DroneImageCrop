CREATE TABLE `analyses` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL,
	`crop_type` text NOT NULL,
	`field_size_hectares` real NOT NULL,
	`average_yield_per_plant_kg` real NOT NULL,
	`plant_count` integer NOT NULL,
	`crop_density` real NOT NULL,
	`crop_coverage` real NOT NULL,
	`vegetation_score` real NOT NULL,
	`health_score` real NOT NULL,
	`estimated_yield` real NOT NULL,
	`confidence_score` real NOT NULL,
	`image_path` text
);
