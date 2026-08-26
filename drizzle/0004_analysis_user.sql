ALTER TABLE `analyses` ADD `user_id` integer REFERENCES `users`(`id`);
