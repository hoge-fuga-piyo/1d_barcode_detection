#include "BarcodeDetector5.h"
#include <opencv2/flann.hpp>

std::tuple<std::vector<cv::Rect>, std::vector<std::vector<cv::Point>>> BarcodeDetector5::detectMserRegions(const cv::Mat& gray_image) const {
	cv::Ptr<cv::MSER> mser = cv::MSER::create();
	std::vector<std::vector<cv::Point>> regions;
	std::vector<cv::Rect> mser_bbox;
	mser->detectRegions(gray_image, regions, mser_bbox);

	return std::tuple<std::vector<cv::Rect>, std::vector<std::vector<cv::Point>>>(mser_bbox, regions);
}

std::vector<Bar5> BarcodeDetector5::removeInvalidRegions(const std::vector<Bar5>& bars) const {
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

		// k-meansで2つのクラスタに分けて、2つのクラスタ間の距離が一定以上なら片方のクラスタを外れ値とみなす
		const int cluster_num = 2;
		const cv::TermCriteria criteria(cv::TermCriteria::EPS, 0, 0.01);
		cv::Mat1i labels;
		cv::Mat centroids;
		cv::kmeans(bar_centers, cluster_num, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centroids);

		cv::Point2f center1 = centroids.row(0);
		cv::Point2f center2 = centroids.row(1);

		std::vector<Bar5> new_clustered_bars;
		const double distance = cv::norm(center1 - center2);
		if (distance > largest_bar_length * 1.0) {
			// 要素の少ない方のクラスタは外れ値とみなす
			std::vector<Bar5> bars_cluster0;
			std::vector<Bar5> bars_cluster1;
			for (int i = 0; i < bar_centers.size(); i++) {
				if (labels(i, 0) == 0) {
					bars_cluster0.push_back(clustered_bars.at(i));
				} else {
					bars_cluster1.push_back(clustered_bars.at(i));
				}
			}

			if (bars_cluster0.size() > bars_cluster1.size()) {
				new_clustered_bars = bars_cluster0;
			} else {
				new_clustered_bars = bars_cluster1;
			}
		} else {
			new_clustered_bars = clustered_bars;
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

			//std::cout << "size: " << tmp_bars.size() << ", " << bars1.size() << ", " << bars2.size() << std::endl;

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

std::vector<std::array<cv::Point2f, 4>> BarcodeDetector5::mergeBars(const std::vector<std::vector<Bar5>>& bars) const {
	std::vector<std::array<cv::Point2f, 4>> barcode_corners;
	for (const auto& clustered_bars : bars) {
		std::vector<cv::Point> merged_region;
		for (const auto& bar : clustered_bars) {
			std::vector<cv::Point> region = bar.getRegion();
			merged_region.insert(merged_region.end(), region.begin(), region.end());
		}
		cv::RotatedRect rotated_rect = cv::minAreaRect(merged_region);
		cv::Point2f corner[4];
		rotated_rect.points(corner);
		std::array<cv::Point2f, 4> barcode_corner{
			corner[0],
			corner[1],
			corner[2],
			corner[3]
		};
		barcode_corners.push_back(barcode_corner);
	}

	return barcode_corners;
}

void BarcodeDetector5::detect(const cv::Mat& image) const {
	bool is_draw_image = true;

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
	std::cout << "remove invalid region: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	std::cout << "MSER box num2: " << bars.size() << std::endl;

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
	std::vector<std::array<cv::Point2f, 4>> barcode_corners = mergeBars(clustered_bars);
	end = std::chrono::system_clock::now();
	std::cout << "barcode region: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	if (is_draw_image) {
		cv::Mat draw_image = image.clone();
		for (const auto& corner : barcode_corners) {
			cv::line(draw_image, corner.at(0), corner.at(1), cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner.at(1), corner.at(2), cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner.at(2), corner.at(3), cv::Scalar(0, 0, 255), 2);
			cv::line(draw_image, corner.at(3), corner.at(0), cv::Scalar(0, 0, 255), 2);
		}
		cv::imshow("barcode", draw_image);
	}
}

BarcodeDetector5::BarcodeDetector5(): min_barcode_bar_num(5) {}
