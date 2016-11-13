package cifar10.classifier.saif.cifar10.retrofit;

import java.util.List;

/**
 * Created by shams on 11/13/2016.
 */

public class ImagesListResponse {
    public List<String> images_list;

    public ImagesListResponse(List<String> images_list) {
        this.images_list = images_list;
    }
}
