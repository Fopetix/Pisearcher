use std::time::Instant;
use std::fs;
use std::error::Error;
use image::{imageops::FilterType, Rgb, RgbImage};

fn hamming_distance(first: &str, second: &str) -> f64 {
    if first.len() != second.len() {
        return 0.0;
    }
    let dif = first.as_bytes()
        .iter()
        .zip(second.as_bytes())
        .filter(|(a, b)| a != b)
        .count();
    1.0 - (dif as f64 / first.len() as f64)
}

type MyResult<T> = std::result::Result<T, Box<dyn Error>>;

use rayon::prelude::*; // Add this at the top with other imports
use std::time::{Duration};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn search_best_match<'a>(searched: &str, text: &'a str) -> MyResult<&'a str> {
    if searched.len() > text.len() {
        return Err("Searched string is longer than text".into());
    }

    let searched_len = searched.len();
    let text_len = text.len();
    let mut total_positions = text_len.saturating_sub(searched_len) + 1;
    let progress_counter = Arc::new(AtomicUsize::new(0));

    // Print progress every 2 seconds
    let progress_thread = {
        let progress_counter = Arc::clone(&progress_counter);
        std::thread::spawn(move || {
            let start_time = Instant::now();
            let mut last_print = Instant::now();
            loop {
                if last_print.elapsed() > Duration::from_secs(2) {
                    let processed = progress_counter.load(Ordering::Relaxed);
                    let progress = (processed as f64 / total_positions as f64) * 100.0;
                    println!("Search progress: {:.1}% ({}/{} positions)",
                             progress, processed, total_positions);
                    last_print = Instant::now();
                }

                if progress_counter.load(Ordering::Relaxed) >= total_positions {
                    break;
                }

                std::thread::sleep(Duration::from_millis(1000));
            }
            println!("Search completed in {:.2?}", start_time.elapsed());
        })
    };
    //total_positions = 10^6;
    let (best_index, best_score) = (0..total_positions)
        .into_par_iter()
        .map(|i| {
            let end = i + searched_len;
            let slice = &text[i..end];
            progress_counter.fetch_add(1, Ordering::Relaxed);
            (i, hamming_distance(searched, slice))
        })
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, 0.0));

    // Ensure progress thread finishes
    progress_counter.store(total_positions, Ordering::Relaxed);
    progress_thread.join().unwrap();

    println!("Best match at index {} with similarity: {:.2}%",
             best_index, best_score * 100.0);

    Ok(&text[best_index..best_index+searched_len])
}

fn img_to_binary_string(filepath: &str, img_size: (u32, u32)) -> MyResult<String> {
    let img = image::open(filepath)?;
    let resized = img.resize_exact(img_size.0, img_size.1, FilterType::Lanczos3);
    let rgb_img = resized.to_rgb8();

    let mut binary_string = String::with_capacity((img_size.0 * img_size.1) as usize);
    for y in 0..img_size.1 {
        for x in 0..img_size.0 {
            let pixel = rgb_img.get_pixel(x, y);
            let total = pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16;
            binary_string.push(if total > 100 { '1' } else { '0' });
        }
    }

    Ok(binary_string)
}

fn binary_string_to_img(img_size: (u32, u32), binary_data: &str,filename: &str) -> MyResult<()> {
    if (img_size.0 * img_size.1) as usize != binary_data.len() {
        return Err("Binary data length doesn't match image dimensions".into());
    }

    let mut img = RgbImage::new(img_size.0, img_size.1);
    let mut chars = binary_data.chars();

    for y in 0..img_size.1 {
        for x in 0..img_size.0 {
            let pixel = match chars.next() {
                Some('1') => Rgb([255, 255, 255]),
                _ => Rgb([0, 0, 0]),
            };
            img.put_pixel(x, y, pixel);
        }
    }

    img.save(filename)?;
    Ok(())
}

fn main() -> MyResult<()> {
    // Configuration
    let image_path = "test.jpg";
    let text_file_path = "output.txt";
    let output_image_size = (16, 12);

    // Step 1: Convert image to binary string
    let image_binary = img_to_binary_string(image_path, output_image_size)?;
    println!("Image converted to binary string (length: {})", image_binary.len());

    // Step 2: Search for closest match in text file
    let file_contents = fs::read_to_string(text_file_path)?;

    let best_match = search_best_match(&image_binary, &file_contents)?;

    // Step 3: Convert the found string back to an image
    binary_string_to_img(output_image_size, &image_binary,"Scaled_down_og.png")?;
    binary_string_to_img(output_image_size, best_match,"pi_replication.png")?;
    println!("Best match converted to output.png");

    Ok(())
}