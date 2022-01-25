// Generated by gencpp from file custom_msgs/ClearLaserRequest.msg
// DO NOT EDIT!


#ifndef CUSTOM_MSGS_MESSAGE_CLEARLASERREQUEST_H
#define CUSTOM_MSGS_MESSAGE_CLEARLASERREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <sensor_msgs/LaserScan.h>

namespace custom_msgs
{
template <class ContainerAllocator>
struct ClearLaserRequest_
{
  typedef ClearLaserRequest_<ContainerAllocator> Type;

  ClearLaserRequest_()
    : weight(0)
    , laser()  {
    }
  ClearLaserRequest_(const ContainerAllocator& _alloc)
    : weight(0)
    , laser(_alloc)  {
  (void)_alloc;
    }



   typedef int8_t _weight_type;
  _weight_type weight;

   typedef  ::sensor_msgs::LaserScan_<ContainerAllocator>  _laser_type;
  _laser_type laser;





  typedef boost::shared_ptr< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> const> ConstPtr;

}; // struct ClearLaserRequest_

typedef ::custom_msgs::ClearLaserRequest_<std::allocator<void> > ClearLaserRequest;

typedef boost::shared_ptr< ::custom_msgs::ClearLaserRequest > ClearLaserRequestPtr;
typedef boost::shared_ptr< ::custom_msgs::ClearLaserRequest const> ClearLaserRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::custom_msgs::ClearLaserRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::custom_msgs::ClearLaserRequest_<ContainerAllocator1> & lhs, const ::custom_msgs::ClearLaserRequest_<ContainerAllocator2> & rhs)
{
  return lhs.weight == rhs.weight &&
    lhs.laser == rhs.laser;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::custom_msgs::ClearLaserRequest_<ContainerAllocator1> & lhs, const ::custom_msgs::ClearLaserRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace custom_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "00cb7b8b21b02f60fc59fa03cb0dfb8d";
  }

  static const char* value(const ::custom_msgs::ClearLaserRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x00cb7b8b21b02f60ULL;
  static const uint64_t static_value2 = 0xfc59fa03cb0dfb8dULL;
};

template<class ContainerAllocator>
struct DataType< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "custom_msgs/ClearLaserRequest";
  }

  static const char* value(const ::custom_msgs::ClearLaserRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# svr for the ClearCells service\n"
"int8 weight\n"
"sensor_msgs/LaserScan laser\n"
"\n"
"================================================================================\n"
"MSG: sensor_msgs/LaserScan\n"
"# Single scan from a planar laser range-finder\n"
"#\n"
"# If you have another ranging device with different behavior (e.g. a sonar\n"
"# array), please find or create a different message, since applications\n"
"# will make fairly laser-specific assumptions about this data\n"
"\n"
"Header header            # timestamp in the header is the acquisition time of \n"
"                         # the first ray in the scan.\n"
"                         #\n"
"                         # in frame frame_id, angles are measured around \n"
"                         # the positive Z axis (counterclockwise, if Z is up)\n"
"                         # with zero angle being forward along the x axis\n"
"                         \n"
"float32 angle_min        # start angle of the scan [rad]\n"
"float32 angle_max        # end angle of the scan [rad]\n"
"float32 angle_increment  # angular distance between measurements [rad]\n"
"\n"
"float32 time_increment   # time between measurements [seconds] - if your scanner\n"
"                         # is moving, this will be used in interpolating position\n"
"                         # of 3d points\n"
"float32 scan_time        # time between scans [seconds]\n"
"\n"
"float32 range_min        # minimum range value [m]\n"
"float32 range_max        # maximum range value [m]\n"
"\n"
"float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)\n"
"float32[] intensities    # intensity data [device-specific units].  If your\n"
"                         # device does not provide intensities, please leave\n"
"                         # the array empty.\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
;
  }

  static const char* value(const ::custom_msgs::ClearLaserRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.weight);
      stream.next(m.laser);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ClearLaserRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::custom_msgs::ClearLaserRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::custom_msgs::ClearLaserRequest_<ContainerAllocator>& v)
  {
    s << indent << "weight: ";
    Printer<int8_t>::stream(s, indent + "  ", v.weight);
    s << indent << "laser: ";
    s << std::endl;
    Printer< ::sensor_msgs::LaserScan_<ContainerAllocator> >::stream(s, indent + "  ", v.laser);
  }
};

} // namespace message_operations
} // namespace ros

#endif // CUSTOM_MSGS_MESSAGE_CLEARLASERREQUEST_H