package cifar10.classifier.saif.cifar10.retrofit;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Query;

/**
 * Created by shams on 11/11/2016.
 */

public interface ImagesService {
    @GET("find_images")
    Call<ImagesListResponse> findImages(@Query("search_string") String searchString);

    @Multipart
    @POST("classify_image")
    Call<String> classifyImage(@Part MultipartBody.Part image);
}
