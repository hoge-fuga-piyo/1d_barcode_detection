#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDetector5.h"

BarcodeDetector5::BarcodeDetector5(): min_barcode_bar_num(5) {}

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

		// �A�X�y�N�g�䂪���炩�Ƀo�[�R�[�h�̃o�[�łȂ����̂͏��O
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

	// �}��ϐ��𓱏o
	cv::Mat parameter;
	cv::solve(left_hand, right_hand, parameter);

	// ��_�𓱏o
	cv::Point2f intersection_point(vertical_vector[0] * parameter.at<double>(0, 0) + center.x, vertical_vector[1] * parameter.at<double>(0, 0) + center.y);

	return intersection_point;
}

std::vector<std::vector<Bar5>> BarcodeDetector5::clustering(const std::vector<Bar5>& bars) const {
	// �e�o�[�ɑ΂��Đ����Ńo�[�̒�����ʂ钼�������߂�
	// �o�[�Ɠ��������ō��W���_��ʂ钼�������߂�
	// ����2�̒����̌�_�́A�����o�[�R�[�h�ɑ�����o�[�ł͂قړ���ɂȂ�͂��Ȃ̂ŁA���̌�_�ɑ΂��ăN���X�^�����O���s��
	std::vector<cv::Point2f> representation_points;
	for (const auto& bar : bars) {
		cv::Point2f intersection_point = conputeRepresentationPoint(bar);
		representation_points.push_back(intersection_point);
	}

	// �����̃N���X�^���쐬
	std::vector<TreeElement> clusters;
	for (int i = 0; i < representation_points.size(); i++) {
		TreeElement elem;
		elem.parent_index = -1;
		elem.point = representation_points.at(i);
		elem.indexes = std::vector<int>{ i };
		clusters.push_back(elem);
	}

	// �N���X�^�����O
	// �P���ȊK�w�I�N���X�^�����O���ƌv�Z�R�X�g�������̂ŁAANN�Ƃ��g����������������
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

		// �N���X�^��1�ɂȂ���
		if (min_pair_index[0] < 0) {
			break;
		}

		// �ŏ��̃N���X�^�ԋ�����臒l�𒴂���
		if (min_distance > cluster_distance_threshold) {
			break;
		}

		TreeElement new_cluster;
		new_cluster.parent_index = -1;
		std::vector<int> new_indexes = clusters.at(min_pair_index[0]).indexes;
		new_indexes.insert(new_indexes.begin(), clusters.at(min_pair_index[1]).indexes.begin(), clusters.at(min_pair_index[1]).indexes.end());
		new_cluster.indexes = new_indexes;

		// �N���X�^�̑�\�_�͂Ƃ肠�����Q���ϖ@�ŋ��߂Ă݂�
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

		// k-means��2�̃N���X�^�ɕ����āA2�̃N���X�^���̓_�̍ŏ����������ȏ�Ȃ�Е��̃N���X�^���O��l�Ƃ݂Ȃ�
		const int cluster_num = 2;
		const cv::TermCriteria criteria(cv::TermCriteria::EPS, 0, 0.01);
		cv::Mat1i labels;
		cv::Mat2d centroids;
		cv::kmeans(bar_centers, cluster_num, labels, criteria, 5, cv::KMEANS_PP_CENTERS, centroids);
		//cv::kmeans(bar_centers, cluster_num, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centroids);

		cv::Point2f center1(centroids.at<double>(0, 0), centroids.at<double>(0, 1));
		cv::Point2f center2(centroids.at<double>(1, 0), centroids.at<double>(1, 1));

		// �N���X�^���Ƃɕ�����
		std::vector<Bar5> bars_cluster0;
		std::vector<Bar5> bars_cluster1;
		for (int i = 0; i < bar_centers.size(); i++) {
			if (labels(i, 0) == 0) {
				bars_cluster0.push_back(clustered_bars.at(i));
			} else {
				bars_cluster1.push_back(clustered_bars.at(i));
			}
		}
		
		// �ŏ����������ȏォ�ǂ����̔���
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

		// �Е��̃N���X�^�͊O��l
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

std::vector<cv::RotatedRect> BarcodeDetector5::concatBarcodes(const std::vector<cv::RotatedRect>& barcodes, const std::vector<std::vector<Bar5>>& bars) const {
	// �o�[�R�[�h�̕����ƍ����ƒ��S�𓱏o
	std::vector<cv::Vec2f> barcode_vectors;
	std::vector<double> barcode_heights;
	std::vector<cv::Point2f> barcode_centers;
	for (int i = 0; i < barcodes.size(); i++) {
		cv::Point2f corner[4];
		barcodes[i].points(corner);
		const cv::Vec2f vector1 = corner[0] - corner[1]; // topLeft to bottomLeft
		const cv::Vec2f vector2 = corner[2] - corner[1]; // topLeft to topRight

		// ����
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

		// ����
		const double height = radian1 > radian2 ? cv::norm(vector1) : cv::norm(vector2);
		barcode_heights.push_back(height);

		// ���S
		const cv::Point2f center = (corner[0] + corner[2]) * 0.5;
		barcode_centers.push_back(center);
	}

	// �o�[�R�[�h�̌���
	std::vector<int> concat_map(barcodes.size(), -1);
	int new_barcode_index = 0;
	for (int i = 0; i < barcodes.size(); i++) {
		bool find_new_concat = false;
		for (int j = i + 1; j < barcodes.size(); j++) {
			// ���Ɍ�������ς݂Ȃ�X�L�b�v
			if (concat_map.at(i) > 0 && concat_map.at(j)) {
				continue;
			}

			// �o�[�R�[�h���m�̍����̍������ȏ�Ȃ猋�����Ȃ�
			const double large_height = barcode_heights.at(i) > barcode_heights.at(j) ? barcode_heights.at(i) : barcode_heights.at(j);
			const double short_height = barcode_heights.at(i) > barcode_heights.at(j) ? barcode_heights.at(j) : barcode_heights.at(i);
			if (short_height / large_height < 0.7) {
				continue;
			}

			// �o�[�R�[�h���m�̊p�x�̍������ȏ�Ȃ猋�����Ȃ�
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

			// �Е��̃o�[�R�[�h�̒��_��������Е��̃o�[�R�[�h�̒��_�ւ̃x�N�g���ƁA���ꂼ��̃o�[�R�[�h�̌��������ȏ�قȂ�Ό������Ȃ�
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

			// �o�[�R�[�h�̗̈悪���ȏ㉓����Ό������Ȃ�
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

	if (barcodes.size() == new_clustered_bars.size()) {	// �������ꂽ�o�[�R�[�h�����݂��Ȃ�
		return barcodes;
	}

	// TODO ��`�̓��o�͌��������o�[�R�[�h�̂��̂����ɍi��Όv�Z�R�X�g��}����ꂻ��
	std::vector<cv::RotatedRect> new_barcodes =  mergeBars(new_clustered_bars);
	if (new_barcodes.size() == barcodes.size()) {
		return new_barcodes;
	}

	return concatBarcodes(new_barcodes, new_clustered_bars);
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
	
	// �o�[�̍\�z
	start = std::chrono::system_clock::now();
	std::vector<Bar5> bars;
	for (size_t i = 0; i < mser_regions.size(); i++) {
		Bar5 bar(mser_bbox.at(i), mser_regions.at(i));
		bars.push_back(bar);
	}
	end = std::chrono::system_clock::now();
	std::cout << "struct bars: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	// �s�v�ȗ̈���폜
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

	// �o�[�ɑ΂��Đ����ȃx�N�g����`�悵�Ă݂�
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

	// �o�[���N���X�^�����O
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

	// �e�N���X�^����O��l������
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

	// �o�[�R�[�h�̗̈�����߂�
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

	// �o�[�R�[�h����������
	start = std::chrono::system_clock::now();
	barcode_rect = concatBarcodes(barcode_rect, clustered_bars);
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

}
