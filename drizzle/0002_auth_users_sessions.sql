CREATE TABLE `users` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`email` text NOT NULL UNIQUE,
	`username` text DEFAULT 'Farmer' NOT NULL,
	`occupation` text DEFAULT 'Agricultural Researcher' NOT NULL,
	`password_hash` text NOT NULL,
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE TABLE `sessions` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`token` text NOT NULL UNIQUE,
	`user_id` integer NOT NULL REFERENCES `users`(`id`),
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL
);
