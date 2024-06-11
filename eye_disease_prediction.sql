-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 11, 2024 at 06:28 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `eye_disease_prediction`
--

-- --------------------------------------------------------

--
-- Table structure for table `prediction`
--

CREATE TABLE `prediction` (
  `id` int(11) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `prediction` varchar(50) NOT NULL,
  `upload_date` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `prediction`
--

INSERT INTO `prediction` (`id`, `filename`, `prediction`, `upload_date`) VALUES
(1, 'cataract_061.png', 'cataract', '2024-06-10 22:50:54'),
(3, '3386_left.jpg', 'normal', '2024-06-10 23:09:03'),
(5, 'images.jpeg', 'glaucoma', '2024-06-10 23:24:12'),
(6, 'depositphotos_568325816-stock-photo-proliferative-diabetic-retinopathy-illustration-showing.jpg', 'retinopathy', '2024-06-10 23:24:27'),
(8, 'Screenshot_2024-06-10_204847.jpg', 'cataract', '2024-06-11 16:54:35'),
(9, 'cataract_061.png', 'cataract', '2024-06-11 16:54:48');

-- --------------------------------------------------------

--
-- Table structure for table `predictions`
--

CREATE TABLE `predictions` (
  `id` int(11) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `prediction` varchar(50) NOT NULL,
  `upload_date` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `prediction`
--
ALTER TABLE `prediction`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `predictions`
--
ALTER TABLE `predictions`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `prediction`
--
ALTER TABLE `prediction`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- AUTO_INCREMENT for table `predictions`
--
ALTER TABLE `predictions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
