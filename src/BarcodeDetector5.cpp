#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDetector5.h"

BarcodeDetector5::BarcodeDetector5(): min_barcode_bar_num(5) {}

std::tuple<std::vector<cv::Rect>, std::vector<std::vector<cv::Point>>> BarcodeDetector5::detectMserRegions(const cv::Mat& gray_image) const {
	cv::Ptr<cv::MSER> mser = cv::MSER::create(5, 60, 5000);
	std::vector<std::vector<cv::Point>> regions;
	std::vector<cv::Rect> mser_bbox;
	mser->detectRegions(gray_image, regions, mser_bbox);

	return std::tuple<std::vector<cv::Rect>, std::vector<std::vector<cv::Point>>>(mser_bbox, regions);
}

std::vector<Bar5> BarcodeDetector5::removeInvalidAspectRatioRegions(const std::vector<Bar5>& bars) const {
	std::vector<Bar5> dst_bars;
	for (const auto& bar : bars) {
		const double aspect_ratio = bar.getAspectRatio();

		// アスペクト比が明らかにバーコードのバーでないものは除外
		if (aspect_ratio < 10.0) {
			continue;
		}

		dst_bars.push_back(bar);
	}

	return dst_bars;
}

std::vector<Bar5> BarcodeDetector5::uniqueSameAreaRegions(const std::vector<Bar5>& bars) const {
	std::vector<Bar5> tmp_bars = bars;
	std::sort(tmp_bars.begin(), tmp_bars.end(), [](const auto& bar1, const auto& bar2) {
		return bar1.getLength() > bar2.getLength();
	});

	std::vector<bool> is_use_list(tmp_bars.size(), true);
	const double distance_threshold = 1.0;
	for (size_t i = 0; i < tmp_bars.size(); i++) {
		const double length1 = tmp_bars.at(i).getLength();
		const std::array<cv::Point2f, 4> corner1 = tmp_bars.at(i).getCorner();
		for (size_t j = i + 1; j < tmp_bars.size(); j++) {
			const double length2 = tmp_bars.at(j).getLength();
			const std::array<cv::Point2f, 4> corner2 = tmp_bars.at(j).getCorner();
			if (length2 / length1 < 0.95) {
				break;
			}

			if (cv::norm(corner1[0] - corner2[0]) < distance_threshold
				&& cv::norm(corner1[1] - corner2[1]) < distance_threshold
				&& cv::norm(corner1[2] - corner2[2]) < distance_threshold
				&& cv::norm(corner1[3] - corner2[3]) < distance_threshold) {
				is_use_list[j] = false;
			}
		}
	}

	std::vector<Bar5> dst_bars;
	for (size_t i = 0; i < is_use_list.size(); i++) {
		if (is_use_list.at(i)) {
			dst_bars.push_back(tmp_bars.at(i));
		}
	}

	return dst_bars;
}

std::vector<Bar5> BarcodeDetector5::removeInvalidRegions(const std::vector<Bar5>& bars) const {
	std::vector<Bar5> dst_bars = removeInvalidAspectRatioRegions(bars);

	// 後続処理の計算量を減らすため、同じ領域を指していると思われるバーをユニークにする
	dst_bars = uniqueSameAreaRegions(dst_bars);

	return dst_bars;
}

cv::Point2f BarcodeDetector5::conputeRepresentationPoint(const Bar5& bar) const {
	const cv::Point2f center = bar.getCenter();
	const cv::Vec2f vertical_vector = bar.getVerticalVector();

	const cv::Point2f origin(0.0, 0.0);
	const cv::Vec2f bar_direction_vector = bar.getBarDirectionVector();
	
	cv::Mat left_hand = (cv::Mat_<double>(2, 2) << vertical_vector[0], -bar_direction_vector[0]
		, vertical_vector[1], -bar_direction_vector[1]);
	cv::Mat right_hand = (cv::Mat_<double>(2, 1) << origin.x - center.x, origin.y - center.y);

	// 媒介変数を導出
	cv::Mat parameter;
	cv::solve(left_hand, right_hand, parameter);

	// 交点を導出
	cv::Point2f intersection_point(vertical_vector[0] * parameter.at<double>(0, 0) + center.x, vertical_vector[1] * parameter.at<double>(0, 0) + center.y);

	return intersection_point;
}

std::vector<std::vector<Bar5>> BarcodeDetector5::clustering(const std::vector<Bar5>& bars) const {
	// 各バーに対して垂直でバーの中央を通る直線を求める
	// バーと同じ方向で座標原点を通る直線を求める
	// この2つの直線の交点は、同じバーコードに属するバーではほぼ同一になるはずなので、この交点に対してクラスタリングを行う
	std::vector<cv::Point2f> representation_points;
	for (const auto& bar : bars) {
		cv::Point2f intersection_point = conputeRepresentationPoint(bar);
		representation_points.push_back(intersection_point);
	}

	// 初期のクラスタを作成
	std::vector<TreeElement> clusters;
	for (int i = 0; i < representation_points.size(); i++) {
		TreeElement elem;
		elem.parent_index = -1;
		elem.point = representation_points.at(i);
		elem.indexes = std::vector<int>{ i };
		clusters.push_back(elem);
	}

	// クラスタリング
	// 単純な階層的クラスタリングだと計算コストが高いので、ANNとか使った方がいいかも
	const double cluster_distance_threshold = 30.0;
	while (true) {
		double min_distance = cluster_distance_threshold + 0.01;
		std::array<int, 2> min_pair_index{ -1, -1 };
		for (int i = 0; i < clusters.size() - 1; i++) {
			if (clusters.at(i).parent_index >= 0) {
				continue;
			}

			for (int j = i + 1; j < clusters.size(); j++) {
				if (clusters.at(j).parent_index >= 0) {
					continue;
				}
				double distance = cv::norm(clusters.at(i).point - clusters.at(j).point);

				if (min_distance > distance) {
					min_distance = distance;
					min_pair_index[0] = i;
					min_pair_index[1] = j;
				}
			}
		}

		// クラスタが1つになった
		if (min_pair_index[0] < 0) {
			break;
		}

		// 最小のクラスタ間距離が閾値を超えた
		if (min_distance > cluster_distance_threshold) {
			break;
		}

		TreeElement new_cluster;
		new_cluster.parent_index = -1;
		std::vector<int> new_indexes = clusters.at(min_pair_index[0]).indexes;
		new_indexes.insert(new_indexes.begin(), clusters.at(min_pair_index[1]).indexes.begin(), clusters.at(min_pair_index[1]).indexes.end());
		new_cluster.indexes = new_indexes;

		// クラスタの代表点はとりあえず群平均法で求めてみる
		TreeElement min_elem1 = clusters.at(min_pair_index[0]);
		TreeElement min_elem2 = clusters.at(min_pair_index[1]);
		new_cluster.point = (min_elem1.point * (float)min_elem1.indexes.size() + min_elem2.point * (float)min_elem2.indexes.size()) / (float)(min_elem1.indexes.size() + min_elem2.indexes.size());

		clusters.push_back(new_cluster);
		clusters[min_pair_index[0]].parent_index = clusters.size() - 1;
		clusters[min_pair_index[1]].parent_index = clusters.size() - 1;
	}

	std::vector<std::vector<Bar5>> clustered_bars;
	for (const auto& cluster : clusters) {
		if (cluster.parent_index >= 0) {
			continue;
		}

		if (cluster.indexes.size() < min_barcode_bar_num) {
			continue;
		}

		std::vector<Bar5> one_clustered_bars;
		for (const int index : cluster.indexes) {
			one_clustered_bars.push_back(bars.at(index));
		}

		clustered_bars.push_back(one_clustered_bars);
	}

	return clustered_bars;
}

std::vector<std::vector<Bar5>> BarcodeDetector5::removeOutlierPositionBars(const std::vector<std::vector<Bar5>>& bars) const {
	std::vector<std::vector<Bar5>> dst_bars;
	for (const auto& clustered_bars : bars) {
		const double largest_bar_length = (*std::max_element(clustered_bars.begin(), clustered_bars.end(), [](const auto& bar1, const auto& bar2) {
			return bar1.getLength() < bar2.getLength();
		})).getLength();

		std::vector<cv::Point2f> bar_centers;
		for (const auto& bar : clustered_bars) {
			bar_centers.push_back(bar.getCenter());
		}

		// k-meansで2つのクラスタに分けて、2つのクラスタ内の点の最小距離が一定以上なら片方のクラスタを外れ値とみなす
		const int cluster_num = 2;
		const cv::TermCriteria criteria(cv::TermCriteria::EPS, 0, 0.01);
		cv::Mat1i labels;
		cv::Mat2d centroids;
		cv::kmeans(bar_centers, cluster_num, labels, criteria, 5, cv::KMEANS_PP_CENTERS, centroids);
		//cv::kmeans(bar_centers, cluster_num, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centroids);

		cv::Point2f center1(centroids.at<double>(0, 0), centroids.at<double>(0, 1));
		cv::Point2f center2(centroids.at<double>(1, 0), centroids.at<double>(1, 1));

		// クラスタごとに分ける
		std::vector<Bar5> bars_cluster0;
		std::vector<Bar5> bars_cluster1;
		for (int i = 0; i < bar_centers.size(); i++) {
			if (labels(i, 0) == 0) {
				bars_cluster0.push_back(clustered_bars.at(i));
			} else {
				bars_cluster1.push_back(clustered_bars.at(i));
			}
		}
		
		// 最小距離が一定以上かどうかの判定
		bool is_far = true;
		for (const auto& bar1 : bars_cluster0) {
			for (const auto& bar2 : bars_cluster1) {
				const double distance = cv::norm(bar1.getCenter() - bar2.getCenter());
				if (distance < largest_bar_length * 1.0) {
					is_far = false;
					break;
				}
			}

			if (!is_far) {
				break;
			}
		}

		// 片方のクラスタは外れ値
		std::vector<Bar5> new_clustered_bars;
		if (is_far) {
			if (bars_cluster0.size() > bars_cluster1.size()) {
				new_clustered_bars = bars_cluster0;
			} else {
				new_clustered_bars = bars_cluster1;
			}
		} else {
			new_clustered_bars = clustered_bars;
		}

		if (new_clustered_bars.size() < min_barcode_bar_num) {
			continue;
		}

		dst_bars.push_back(new_clustered_bars);
	}

	return dst_bars;
}

std::vector<std::vector<Bar5>> BarcodeDetector5::removeOutlierLengthBars(const std::vector<std::vector<Bar5>>& bars) const {
	std::vector<std::vector<Bar5>> dst_bars;
	for (const auto& clustered_bars : bars) {
		std::vector<Bar5> tmp_bars = clustered_bars;
		std::sort(tmp_bars.begin(), tmp_bars.end(), [](const auto& bar1, const auto& bar2) {
			return bar1.getLength() < bar2.getLength();
		});

		int split_index = -1;
		const double largest_bar_length = tmp_bars.at(tmp_bars.size() - 1).getLength();
		for (size_t i = 0; i < tmp_bars.size(); i++) {
			const double length = tmp_bars.at(i).getLength();
			//std::cout << largest_bar_length << ", " << length << ", "<<length/largest_bar_length << std::endl;
			if (length / largest_bar_length > 0.67) {
				break;
			} else {
				split_index = i;
			}
		}

		if (split_index < 0) {
			dst_bars.push_back(tmp_bars);
		} else {
			std::vector<Bar5> bars1{ tmp_bars.begin(), tmp_bars.begin() + split_index + 1 };
			std::vector<Bar5> bars2{ tmp_bars.begin() + split_index + 1, tmp_bars.end() };

			if (bars1.size() >= min_barcode_bar_num) {
				dst_bars.push_back(bars1);
			}
			if (bars2.size() >= min_barcode_bar_num) {
				dst_bars.push_back(bars2);
			}
		}
	}

	return dst_bars;

}

std::vector<std::vector<Bar5>> BarcodeDetector5::removeInvalidBars(const std::vector<std::vector<Bar5>>& bars) const {
	std::vector<std::vector<Bar5>> dst_bars = removeOutlierLengthBars(bars);
	dst_bars = removeOutlierPositionBars(dst_bars);

	return dst_bars;
}

std::vector<cv::RotatedRect> BarcodeDetector5::mergeBars(const std::vector<std::vector<Bar5>>& bars) const {
	std::vector<std::array<cv::Point2f, 4>> barcode_corners;
	std::vector<cv::RotatedRect> barcode_rect;
	for (const auto& clustered_bars : bars) {
		std::vector<cv::Point> merged_region;
		for (const auto& bar : clustered_bars) {
			std::vector<cv::Point> region = bar.getRegion();
			merged_region.insert(merged_region.end(), region.begin(), region.end());
		}
		cv::RotatedRect rotated_rect = cv::minAreaRect(merged_region);
		barcode_rect.push_back(rotated_rect);
	}

	return barcode_rect;
}

std::tuple<std::vector<cv::RotatedRect>, std::vector<std::vector<Bar5>>> BarcodeDetector5::concatBarcodes(const std::vector<cv::RotatedRect>& barcodes, const std::vector<std::vector<Bar5>>& bars) const {
	// バーコードの方向と高さと中心を導出
	std::vector<cv::Vec2f> barcode_vectors;
	std::vector<double> barcode_heights;
	std::vector<cv::Point2f> barcode_centers;
	for (int i = 0; i < barcodes.size(); i++) {
		cv::Point2f corner[4];
		barcodes[i].points(corner);
		const cv::Vec2f vector1 = corner[0] - corner[1]; // topLeft to bottomLeft
		const cv::Vec2f vector2 = corner[2] - corner[1]; // topLeft to topRight

		// 方向
		const cv::Vec2f bar_vertical_vector = bars.at(i).at(0).getVerticalVector();
		const double cos_theta1 = bar_vertical_vector.dot(vector1) / (cv::norm(bar_vertical_vector) * cv::norm(vector1));
		double radian1 = std::acos(cos_theta1);
		if (radian1 > M_PI / 2.0) {
			radian1 = M_PI - radian1;
		}
		const double cos_theta2 = bar_vertical_vector.dot(vector2) / (cv::norm(bar_vertical_vector) * cv::norm(vector2));
		double radian2 = std::acos(cos_theta2);
		if (radian2 > M_PI / 2.0) {
			radian2 = M_PI - radian2;
		}

		const cv::Vec2f barcode_vector = radian1 > radian2 ? vector2 : vector1;
		barcode_vectors.push_back(barcode_vector);

		// 高さ
		const double height = radian1 > radian2 ? cv::norm(vector1) : cv::norm(vector2);
		barcode_heights.push_back(height);

		// 中心
		const cv::Point2f center = (corner[0] + corner[2]) * 0.5;
		barcode_centers.push_back(center);
	}

	// バーコードの結合
	std::vector<int> concat_map(barcodes.size(), -1);
	int new_barcode_index = 0;
	for (int i = 0; i < barcodes.size(); i++) {
		bool find_new_concat = false;
		for (int j = i + 1; j < barcodes.size(); j++) {
			// 既に結合判定済みならスキップ
			if (concat_map.at(i) > 0 && concat_map.at(j)) {
				continue;
			}

			// バーコード同士の高さの差が一定以上なら結合しない
			const double large_height = barcode_heights.at(i) > barcode_heights.at(j) ? barcode_heights.at(i) : barcode_heights.at(j);
			const double short_height = barcode_heights.at(i) > barcode_heights.at(j) ? barcode_heights.at(j) : barcode_heights.at(i);
			if (short_height / large_height < 0.7) {
				continue;
			}

			// バーコード同士の角度の差が一定以上なら結合しない
			const double radian_threshold1 = 10.0 * (M_PI / 180.0);
			const cv::Vec2f vector1 = barcode_vectors.at(i);
			const cv::Vec2f vector2 = barcode_vectors.at(j);
			const double cos_theta = vector1.dot(vector2) / (cv::norm(vector1) * cv::norm(vector2));
			double radian = std::acos(cos_theta);
			if (radian > M_PI / 2.0) {
				radian = M_PI - radian;
			}
			if (radian > radian_threshold1) {
				continue;
			}

			// 片方のバーコードの中点からもう片方のバーコードの中点へのベクトルと、それぞれのバーコードの向きが一定以上異なれば結合しない
			const double radian_threshold2 = 10.0 * (M_PI / 180.0);
			const cv::Vec2f center2center = barcode_centers.at(j) - barcode_centers.at(i);
			const double cos_theta1 = center2center.dot(barcode_vectors.at(i)) / (cv::norm(center2center) * cv::norm(barcode_vectors.at(i)));
			double radian1 = std::acos(cos_theta1);
			if (radian1 > M_PI / 2.0) {
				radian1 = M_PI - radian1;
			}
			const double cos_theta2 = center2center.dot(barcode_vectors.at(j)) / (cv::norm(center2center) * cv::norm(barcode_vectors.at(j)));
			double radian2 = std::acos(cos_theta2);
			if (radian2 > M_PI / 2.0) {
				radian2 = M_PI - radian2;
			}
			if (radian1 > radian_threshold2 || radian2 > radian_threshold2) {
				continue;
			}

			// バーコードの領域が一定以上遠ければ結合しない
			bool is_concat_target = false;
			for (const auto& bar1 : bars.at(i)) {
				for (const auto& bar2 : bars.at(j)) {
					const double distance = cv::norm(bar1.getCenter() - bar2.getCenter());
					if (distance < 30.0) {
						is_concat_target = true;
						break;
					}
				}

				if (is_concat_target) {
					break;
				}
			}
			
			if (is_concat_target) {
				if (concat_map.at(i) >= 0) {
					concat_map[j] = concat_map.at(i);
				} else if (concat_map.at(j) >= 0) {
					concat_map[i] = concat_map.at(j);
				} else {
					concat_map[i] = new_barcode_index;
					concat_map[j] = new_barcode_index;
					find_new_concat = true;
				}
			}
		}

		if (find_new_concat) {
			new_barcode_index++;
		}
	}

	for (int concat_index : concat_map) {
		std::cout << concat_index << ", ";
	}
	std::cout << "" << std::endl;

	std::vector<std::vector<Bar5>> new_clustered_bars;
	for (int i = 0; i < new_barcode_index; i++) {
		std::vector<Bar5> new_bars;
		for (int j = 0; j < concat_map.size(); j++) {
			if (i == concat_map.at(j)) {
				new_bars.insert(new_bars.end(), bars.at(j).begin(), bars.at(j).end());
			}
		}
		new_clustered_bars.push_back(new_bars);
	}
	for (int i = 0; i < concat_map.size(); i++) {
		if (concat_map.at(i) < 0) {
			new_clustered_bars.push_back(bars.at(i));
		}
	}

	// TODO 矩形の導出は結合したバーコードのものだけに絞れば計算コストを抑えられそう
	std::vector<cv::RotatedRect> new_barcodes =  mergeBars(new_clustered_bars);
	if (new_clustered_bars.size() == bars.size()) {
		return std::tuple<std::vector<cv::RotatedRect>, std::vector<std::vector<Bar5>>>(new_barcodes, new_clustered_bars);
	}

	return concatBarcodes(new_barcodes, new_clustered_bars);
}

std::tuple<std::vector<cv::RotatedRect>, std::vector<std::vector<Bar5>>> BarcodeDetector5::removeInvalidAspectRatioBarcodes(const std::vector<cv::RotatedRect>& barcodes, const std::vector<std::vector<Bar5>>& bars) const {
	std::vector<cv::RotatedRect> dst_barcodes;
	std::vector<std::vector<Bar5>> dst_bars;
	for (size_t i = 0; i < barcodes.size(); i++) {
		// バーコードの方向
		cv::Point2f corner[4];
		barcodes[i].points(corner);
		const cv::Vec2f vector1 = corner[0] - corner[1]; // topLeft to bottomLeft
		const cv::Vec2f vector2 = corner[2] - corner[1]; // topLeft to topRight

		const cv::Vec2f bar_vertical_vector = bars.at(i).at(0).getVerticalVector();
		const double cos_theta1 = bar_vertical_vector.dot(vector1) / (cv::norm(bar_vertical_vector) * cv::norm(vector1));
		double radian1 = std::acos(cos_theta1);
		if (radian1 > M_PI / 2.0) {
			radian1 = M_PI - radian1;
		}
		const double cos_theta2 = bar_vertical_vector.dot(vector2) / (cv::norm(bar_vertical_vector) * cv::norm(vector2));
		double radian2 = std::acos(cos_theta2);
		if (radian2 > M_PI / 2.0) {
			radian2 = M_PI - radian2;
		}

		const cv::Vec2f barcode_vector = radian1 > radian2 ? vector2 : vector1;
		const cv::Vec2f barcode_vertical_vector = radian1 > radian2 ? vector1 : vector2;

		// バーコードの長さ
		const double barcode_width = cv::norm(barcode_vector);

		// バーコードに向きに垂直な領域の長さ
		const double barcode_height = cv::norm(barcode_vertical_vector);
		
		if (barcode_height / barcode_width < 2.0) {
			dst_barcodes.push_back(barcodes.at(i));
			dst_bars.push_back(bars.at(i));
		}
	}

	return std::tuple<std::vector<cv::RotatedRect>, std::vector<std::vector<Bar5>>>(dst_barcodes, dst_bars);
}

std::tuple<std::vector<cv::RotatedRect>, std::vector<std::vector<Bar5>>> BarcodeDetector5::removeInvalidBarNumBarcodes(const std::vector<cv::RotatedRect>& barcodes, const std::vector<std::vector<Bar5>>& bars) const {
	std::vector<cv::RotatedRect> dst_barcodes;
	std::vector<std::vector<Bar5>> dst_bars;
	for (size_t i = 0; i < barcodes.size(); i++) {
		if (bars.at(i).size () >= 10) {
			dst_barcodes.push_back(barcodes.at(i));
			dst_bars.push_back(bars.at(i));
		}
	}

	return std::tuple<std::vector<cv::RotatedRect>, std::vector<std::vector<Bar5>>>(dst_barcodes, dst_bars);
}

std::tuple<std::vector<cv::RotatedRect>, std::vector<std::vector<Bar5>>> BarcodeDetector5::removeInvalidBarcodes(const std::vector<cv::RotatedRect>& barcodes, const std::vector<std::vector<Bar5>>& bars) const {
	auto dst_barcodes_info = removeInvalidAspectRatioBarcodes(barcodes, bars);
	dst_barcodes_info = removeInvalidBarNumBarcodes(std::get<0>(dst_barcodes_info), std::get<1>(dst_barcodes_info));

	return dst_barcodes_info;
}

std::vector<std::tuple<std::array<cv::Point2f, 4>, cv::Vec2f>> BarcodeDetector5::detect(const cv::Mat& image) const {
	bool is_draw_image = false;

	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	if (is_draw_image) {
		cv::imshow("gray image", gray_image);
	}

	// MSER
	auto start = std::chrono::system_clock::now();
	auto mser_result = detectMserRegions(gray_image);
	std::vector<std::vector<cv::Point>> mser_regions = std::get<1>(mser_result);
	std::vector<cv::Rect> mser_bbox = std::get<0>(mser_result);
	auto end = std::chrono::system_clock::now();
	std::cout << "MSER detection: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	std::cout << "MSER box num: " << mser_bbox.size() << std::endl;

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& rect : mser_bbox) {
			cv::rectangle(draw_image, rect, cv::Scalar(0, 255, 0));
		}

		//for (const auto& region : mser_regions) {
		//	cv::Scalar color(
		//		std::rand() % 256,
		//		std::rand() % 256,
		//		std::rand() % 256
		//	);
		//	for (const auto& point : region) {
		//		cv::circle(draw_image, point, 1, color);
		//	}
		//}
		cv::imshow("mser", draw_image);
	}

	//// 後続の処理時間を減らすため、同じ領域を示していると思われるものは1つに統一する
	//start = std::chrono::system_clock::now();
	//const auto& adjusted_bar_info = removeInvalidRegions(mser_bbox, mser_regions);
	//mser_bbox = std::get<0>(adjusted_bar_info);
	//mser_regions = std::get<1>(adjusted_bar_info);
	//end = std::chrono::system_clock::now();
	//std::cout << "remove invalid region: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	//std::cout << "MSER box num2: " << mser_bbox.size() << std::endl;
	
	// バーの構築
	start = std::chrono::system_clock::now();
	std::vector<Bar5> bars;
	for (size_t i = 0; i < mser_regions.size(); i++) {
		Bar5 bar(mser_bbox.at(i), mser_regions.at(i));
		bars.push_back(bar);
	}
	end = std::chrono::system_clock::now();
	std::cout << "struct bars: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	// 不要な領域を削除
	start = std::chrono::system_clock::now();
	bars = removeInvalidRegions(bars);
	end = std::chrono::system_clock::now();
	std::cout << "remove invalid region2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	std::cout << "MSER box num2: " << bars.size() << std::endl;

	//std::vector<Bar5> tmp_bars = bars;
	//std::sort(tmp_bars.begin(), tmp_bars.end(), [](const auto& bar1, const auto& bar2) {
	//	return bar1.getRegion().size() < bar2.getRegion().size();
	//});
	//for(const auto& bar: tmp_bars) {
	//	std::cout<< bar.getRegion().size()<< std::endl;
	//	std::cout<< "[" << bar.getBox().x << ", " << bar.getBox().y << "] ";
	//	std::cout<< "[" << bar.getBox().width << ", " << bar.getBox().height << "] " << std::endl;
	//	std::cout<< "[" << (bar.getBox().x + bar.getBox().width) * 0.5 << ", " << (bar.getBox().y + bar.getBox().height) * 0.5 << "]" << std::endl;
	//}

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& bar : bars) {
			const cv::Rect rect = bar.getBox();
			cv::rectangle(draw_image, rect, cv::Scalar(0, 255, 0));
		}

		cv::imshow("remove invalid box", draw_image);
	}

	// バーに対して垂直なベクトルを描画してみる
	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& bar : bars) {
			const cv::Rect rect = bar.getBox();
			cv::rectangle(draw_image, rect, cv::Scalar(0, 255, 0));
		}

		for (const auto& bar : bars) {
			const cv::Vec2f vertical_vec = bar.getVerticalVector();
			const cv::Point2f start_point = bar.getCenter();
			const cv::Point2f end_point = start_point + cv::Point2f(vertical_vec[0], vertical_vec[1]) * 20.0;

			cv::line(draw_image, start_point, end_point, cv::Scalar(0, 0, 255));
		}

		cv::imshow("vertical vec", draw_image);
	}

	// バーをクラスタリング
	start = std::chrono::system_clock::now();
	std::vector<std::vector<Bar5>> clustered_bars = clustering(bars);
	end = std::chrono::system_clock::now();
	std::cout << "clustering: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	std::cout << "cluster num: " << clustered_bars.size() << std::endl;

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& one_clustered_ber : clustered_bars) {
			cv::Scalar color(
				std::rand() % 256,
				std::rand() % 256,
				std::rand() % 256
			);

			//std::cout << "==========start cluster===========" << std::endl;
			for (const auto& bar : one_clustered_ber) {
				cv::rectangle(draw_image, bar.getBox(), color);
				//std::cout << bar.getCenter() << std::endl;
			}
			//std::cout << "==========end cluster===========" << std::endl;
		}

		cv::imshow("clustered bars", draw_image);
	}

	// 各クラスタから外れ値を除く
	start = std::chrono::system_clock::now();
	clustered_bars = removeInvalidBars(clustered_bars);
	end = std::chrono::system_clock::now();
	std::cout << "remove invalid bars from cluster: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	std::cout << "cluster num: " << clustered_bars.size() << std::endl;

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		std::srand(0);
		for (const auto& one_clustered_ber : clustered_bars) {
			cv::Scalar color(
				std::rand() % 256,
				std::rand() % 256,
				std::rand() % 256
			);

			//std::cout << "==========start cluster===========" << std::endl;
			for (const auto& bar : one_clustered_ber) {
				cv::rectangle(draw_image, bar.getBox(), color);
				//std::cout << bar.getCenter() << std::endl;
			}
			//std::cout << "==========end cluster===========" << std::endl;
		}

		cv::imshow("clustered bars after removing", draw_image);
	}

	// バーコードの領域を求める
	start = std::chrono::system_clock::now();
	std::vector<cv::RotatedRect> barcode_rect = mergeBars(clustered_bars);
	end = std::chrono::system_clock::now();
	std::cout << "barcode region: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	std::cout << "barcode num: " << barcode_rect.size() << std::endl;

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& rect : barcode_rect) {
			cv::Point2f corner[4];
			rect.points(corner);

			cv::line(draw_image, corner[0], corner[1], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[1], corner[2], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[2], corner[3], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[3], corner[0], cv::Scalar(0, 0, 255), 2);
		}
		cv::imshow("barcode", draw_image);
	}

	// バーコードを結合する
	start = std::chrono::system_clock::now();
	const auto concat_barcodes = concatBarcodes(barcode_rect, clustered_bars);
	barcode_rect = std::get<0>(concat_barcodes);
	clustered_bars = std::get<1>(concat_barcodes);
	end = std::chrono::system_clock::now();
	std::cout << "barcode concat: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	std::cout << "barcode num: " << barcode_rect.size() << std::endl;

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& rect : barcode_rect) {
			cv::Point2f corner[4];
			rect.points(corner);

			cv::line(draw_image, corner[0], corner[1], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[1], corner[2], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[2], corner[3], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[3], corner[0], cv::Scalar(0, 0, 255), 2);
		}
		cv::imshow("barcode concat", draw_image);
	}

	for(const auto& bars : clustered_bars) {
		std::cout << "bar num: " << bars.size() << std::endl;
	}

	// バーコードっぽくないやつは削除する
	start = std::chrono::system_clock::now();
	const auto filtered_barcodes = removeInvalidBarcodes(barcode_rect, clustered_bars);
	barcode_rect = std::get<0>(filtered_barcodes);
	clustered_bars = std::get<1>(filtered_barcodes);
	end = std::chrono::system_clock::now();
	std::cout << "filtered barcode: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	std::cout << "barcode num: " << barcode_rect.size() << std::endl;

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& rect : barcode_rect) {
			cv::Point2f corner[4];
			rect.points(corner);

			cv::line(draw_image, corner[0], corner[1], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[1], corner[2], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[2], corner[3], cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner[3], corner[0], cv::Scalar(0, 0, 255), 2);
		}
		//for (const auto& bars : clustered_bars) {
		//	for (const auto& bar : bars) {
		//		const cv::Rect rect = bar.getBox();
		//		cv::rectangle(draw_image, rect, cv::Scalar(0, 255, 0));

		//		const cv::Vec2f vertical_vec = bar.getVerticalVector();
		//		const cv::Point2f start_point = bar.getCenter();
		//		const cv::Point2f end_point = start_point + cv::Point2f(vertical_vec[0], vertical_vec[1]) * 20.0;
		//		cv::line(draw_image, start_point, end_point, cv::Scalar(0, 255, 255));
		//	}
		//}

		cv::imshow("filtered barcode", draw_image);
	}

	// バーコードのコーナーと向きを返す
	std::vector<std::tuple<std::array<cv::Point2f, 4>, cv::Vec2f>> results;
	for (size_t i = 0; i < barcode_rect.size(); i++) {
		// コーナー
		cv::Point2f corner[4];
		barcode_rect[i].points(corner);
		std::array<cv::Point2f, 4> corner_arr{
			corner[0],
			corner[1],
			corner[2],
			corner[3]
		};

		// 向き
		const cv::Vec2f vector1 = corner[0] - corner[1]; // topLeft to bottomLeft
		const cv::Vec2f vector2 = corner[2] - corner[1]; // topLeft to topRight

		const cv::Vec2f bar_vertical_vector = clustered_bars.at(i).at(0).getVerticalVector();
		const double cos_theta1 = bar_vertical_vector.dot(vector1) / (cv::norm(bar_vertical_vector) * cv::norm(vector1));
		double radian1 = std::acos(cos_theta1);
		if (radian1 > M_PI / 2.0) {
			radian1 = M_PI - radian1;
		}
		const double cos_theta2 = bar_vertical_vector.dot(vector2) / (cv::norm(bar_vertical_vector) * cv::norm(vector2));
		double radian2 = std::acos(cos_theta2);
		if (radian2 > M_PI / 2.0) {
			radian2 = M_PI - radian2;
		}

		const cv::Vec2f barcode_vector = radian1 > radian2 ? vector2 : vector1;

		results.push_back(std::tuple<std::array<cv::Point2f, 4>, cv::Vec2f>(corner_arr, barcode_vector));
	}

	return results;
}